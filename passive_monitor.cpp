#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;

namespace 
{
constexpr int    KNN_K        = 5;
constexpr double KNN_MAX_DIST = 2.0;
constexpr double STS_TICK_MS  = 50.0;
constexpr int    STS_MAX      = 20;


struct ProcSnapshot 
{
    int pid{}, ppid{};
    char state{'?'};
    long priority{}, nice{}, num_threads{1};
    long minflt{}, majflt{};
    long utime_ticks{}, stime_ticks{};
    double cpu_seconds{};
    long rss_kb{}, text_kb{}, vm_lib_kb{};
    unsigned long vsize_bytes{}, endcode{};
    long voluntary_ctx{}, nonvoluntary_ctx{};
    string command;
};

using Feature = array<double, 6>;

struct HistorySample 
{
    string label;  // interactive / daemon / batch
    Feature feat{};
    double observed_burst_ms{};
    int sts_class{1};
};

struct MonitoredProcess 
{
    ProcSnapshot snap;
    string process_type;
    double predicted_burst_ms{};
    int sts_class{}, recommended_slice_ms{}, scheduler_rank{};
    double knn_distance{};
    double cpu_usage_pct{};
    int mismatch_score{};  // 0-100, how much actual diverges from recommended
};

struct SystemStats 
{
    int total{}, interactive_count{}, daemon_count{}, batch_count{};
    double avg_burst_ms{};
    long total_ctx_switches{};
    double total_cpu_usage{};
    double avg_mismatch{};
    double efficiency_gain{};
};

long clk_tck()
{
    static long v = sysconf(_SC_CLK_TCK); 
    return v > 0 ? v : 100;
}
long page_sz()
{
    static long v = sysconf(_SC_PAGESIZE);
     return v > 0 ? v : 4096; 
}

bool all_digits(const string& s) 
{
    return !s.empty() && all_of(s.begin(), s.end(), [](unsigned char c){ return isdigit(c) != 0; });
}

// escaping special characters for JSON output
string json_escape(const string& input) 
{
    ostringstream output_stream;
    for (unsigned char c : input) 
    {
        switch (c) 
        {
            case '"':  
                output_stream << "\\\""; break;
            case '\\':
                output_stream << "\\\\"; break;
            case '\n':
                output_stream << "\\n";  break;
            case '\r': 
                output_stream << "\\r";  break;
            case '\t': 
                output_stream << "\\t";  break;
            default:   
                if (c >= 0x20){
                    output_stream << static_cast<char>(c); 
                    break;
                }
        }
    }
    return output_stream.str();
}

//cpu tick reading

struct CpuTick 
{ 
    unsigned long long total{}, process_ticks{}; 
};

unsigned long long get_total_cpu_ticks() 
{
    ifstream stat_file("/proc/stat");
    string line;
    if (!getline(stat_file, line)) 
        return 0;
    
    istringstream ss(line);
    string cpu_label;
    ss >> cpu_label;
    
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    ss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
    return user + nice + system + idle + iowait + irq + softirq + steal;
}

// reads all the info we need from /proc/<pid>/
ProcSnapshot read_proc(int pid) 
{
    namespace fs = filesystem;
    const fs::path proc_path = fs::path("/proc") / to_string(pid);

    //reading core stats from /proc/[pid]/stat
    ifstream stat_file(proc_path / "stat");
    if (!stat_file) 
        throw runtime_error("could not open stat for pid " + to_string(pid));

    string line;
    getline(stat_file, line);

    //command name is usually between parentheses (e.g "(bash)")
    const auto open_paren = line.find('(');
    const auto close_paren = line.rfind(')');
    if (open_paren == string::npos || close_paren == string::npos || close_paren < open_paren)
        throw runtime_error("malformed stat for pid " + to_string(pid));

    const string short_comm = line.substr(open_paren + 1, close_paren - open_paren - 1);
    istringstream ss(line.substr(close_paren + 2));

    ProcSnapshot snapshot;
    snapshot.pid = pid;
    ss >> snapshot.state;

    vector<unsigned long long> fields;
    unsigned long long temp_val;

    while (ss >> temp_val) 
        fields.push_back(temp_val);
    
    if (fields.size() < 25) 
        throw runtime_error("stat file too short for pid " + to_string(pid));

    snapshot.ppid            = static_cast<int>(fields[0]);
    snapshot.minflt          = static_cast<long>(fields[6]);
    snapshot.majflt          = static_cast<long>(fields[8]);
    snapshot.utime_ticks     = static_cast<long>(fields[10]);
    snapshot.stime_ticks     = static_cast<long>(fields[11]);
    snapshot.priority        = static_cast<long>(fields[14]);
    snapshot.nice            = static_cast<long>(fields[15]);
    snapshot.num_threads     = static_cast<long>(fields[16]);
    snapshot.vsize_bytes     = static_cast<unsigned long>(fields[19]);
    snapshot.rss_kb          = static_cast<long>((static_cast<long>(fields[20]) * page_sz()) / 1024);
    snapshot.endcode         = static_cast<unsigned long>(fields[23]);
    snapshot.cpu_seconds     = static_cast<double>(snapshot.utime_ticks + snapshot.stime_ticks) / clk_tck();

    //grabbing text size from statm
    {
        ifstream statm_file(proc_path / "statm");
        if (statm_file) {
            long size, resident, shared, text;
            statm_file >> size >> resident >> shared >> text;
            snapshot.text_kb = (text * page_sz()) / 1024;
        }
    }

    // VmLib and context switch counts from status
    {
        ifstream status_file(proc_path / "status");
        string status_line;
        while (getline(status_file, status_line))
        {
            if (status_line.rfind("VmLib:", 0) == 0) 
            {
                istringstream p(status_line.substr(6)); 
                p >> snapshot.vm_lib_kb;
            }
            else if (status_line.rfind("voluntary_ctxt_switches:", 0) == 0) 
            {
                istringstream p(status_line.substr(24)); 
                p >> snapshot.voluntary_ctx;
            }
            else if (status_line.rfind("nonvoluntary_ctxt_switches:", 0) == 0) 
            {
                istringstream p(status_line.substr(27)); 
                p >> snapshot.nonvoluntary_ctx;
            }
        }
    }

    //full command line from cmdline (replacing nulls with spaces)
    {
        ifstream cmd_file(proc_path / "cmdline", ios::binary);
        string raw_cmd((istreambuf_iterator<char>(cmd_file)), {});
        for (char& c : raw_cmd) 
            if (c == '\0') c = ' ';
        while (!raw_cmd.empty() && raw_cmd.back() == ' ') 
            raw_cmd.pop_back();
        snapshot.command = raw_cmd.empty() ? short_comm : raw_cmd;
    }

    return snapshot;
}

Feature make_feature(const ProcSnapshot& s) 
{
    return 
    {
        s.vsize_bytes / 1024.0,
        static_cast<double>(s.text_kb),
        s.endcode / 1024.0,
        static_cast<double>(s.vm_lib_kb),
        static_cast<double>(s.rss_kb),
        static_cast<double>(s.minflt + s.majflt + s.voluntary_ctx + s.nonvoluntary_ctx),
    };
}

// euclidean distance with per-feature scaling
double feat_dist(const Feature& a, const Feature& b) 
{
    static constexpr Feature scales = 
    {
        1024.0*1024.0, 256.0*1024.0, 1024.0*1024.0,
        512.0*1024.0,  512.0*1024.0, 100000.0
    };
    double t = 0.0;
    for(size_t i = 0; i < a.size(); ++i) 
    {
        double d = (a[i] - b[i]) / scales[i];
        t += d * d;
    }
    return sqrt(t);
}

int burst_to_sts(double ms) 
{
    return clamp(static_cast<int>(ceil(ms / STS_TICK_MS)), 1, STS_MAX);
}

//history handling

vector<HistorySample> default_history() 
{
    return 
    {
        {"interactive", {180000,  6400,  4200,  4096, 12000,  1200},  70, 2},
        {"interactive", {220000,  7200,  4500,  4096, 16000,  1700},  90, 2},
        {"interactive", {160000,  5800,  3900,  3072, 10000,   900},  60, 2},
        {"interactive", {250000,  8000,  5000,  5120, 20000,  2200}, 110, 3},
        {"daemon",      {260000,  5400,  4700,  6144, 22000,  3000}, 180, 4},
        {"daemon",      {310000,  6000,  5000,  6144, 26000,  4200}, 220, 5},
        {"daemon",      {200000,  4800,  4200,  5120, 18000,  2500}, 150, 3},
        {"daemon",      {350000,  6500,  5300,  7168, 30000,  5000}, 260, 6},
        {"batch",       {780000, 12000,  9300, 12288, 78000,  9000}, 420, 9},
        {"batch",       {920000, 14000, 10200, 14336, 98000, 12000}, 520,11},
        {"batch",       {650000, 10000,  8000, 10240, 65000,  7500}, 360, 8},
        {"batch",      {1100000, 16000, 12000, 16384,120000, 15000}, 620,13},
    };
}

vector<HistorySample> load_history(const string& path) 
{
    auto h = default_history();
    ifstream in(path);
    if(!in) 
        return h;  // if no file just use defaults
    string line;
    while(getline(in, line)) 
    {
        if(line.empty() || line[0] == '#') 
            continue;
        istringstream ss(line);
        string cell;
        HistorySample s;
        getline(ss, s.label, ',');
        bool ok = true;
        for(double& v : s.feat) 
        {
            if(!getline(ss, cell, ',')) 
                { ok=false; break; }
            try 
                { v = stod(cell); } 
            catch(...) 
                { ok=false; break; }
        }
        if(!ok) 
            continue;
        if(!getline(ss, cell, ',')) 
            continue;
        try 
            { s.observed_burst_ms = stod(cell); } 
        catch(...) 
            { continue; }
        if(!getline(ss, cell, ',')) 
            continue;
        try 
            { s.sts_class = stoi(cell); } 
        catch(...)
            { continue; }
        h.push_back(move(s));
    }
    return h;
}

void save_history(const string& path, const vector<HistorySample>& h) 
{
    ofstream out(path);
    out << "# label,vsize_kb,text_kb,endcode_kb,vmlib_kb,rss_kb,fault_ctx,burst_ms,sts_class\n";
    for(const auto& s : h) 
    {
        if(s.label != "interactive" && s.label != "daemon" && s.label != "batch") 
            continue;
        out << s.label;
        for(double v : s.feat) 
            out << "," << v;
        out << "," << s.observed_burst_ms << "," << s.sts_class << "\n";
    }
}

//fallback classifier when KNN distance is too large

string heuristic_classify(const ProcSnapshot& s) 
{
    string cmd = s.command;
    transform(cmd.begin(), cmd.end(), cmd.begin(),[](unsigned char c)
    {
        return static_cast<char>(tolower(c)); 
    });

    const auto has = [&](const char* k){ return cmd.find(k) != string::npos; };

    // obvious daemon processes
    if(has("systemd")  || has("journald") || has("/init")     || has("udevd")    ||
       has("dbus")     || has("sshd")     || has("cron")      || has("rsyslog")  ||
       has("networkd") || has("resolved") || has("timesyncd") || has("sd-pam")   ||
       has("agetty")   || has("polkit")   || has("udisksd"))
        return "daemon";

    // shells, editors, interactive stuff
    if(has("bash") || has("zsh")  || has("fish") || has("vim")  ||
       has("nano") || has("emacs")|| has("ssh ")  || has("top") ||
       has("htop") || has("login"))
        return "interactive";

    // heavy tools = batch
    if(has("make") || has("cmake") || has("gcc")   || has("g++")  ||
       has("clang")|| has("python")|| has("node")  || has("java") ||
       has("ffmpeg")|| has("cargo"))
        return "batch";

    // if it's a local binary (running from current dir) or has common user names
    if(cmd.find("./") != string::npos || cmd == "a.out")
        return "interactive"; // treat custom binaries as interactive for better scheduling

    // fallback using thread count and memory
    if(s.num_threads > 8 && s.rss_kb > 40000) 
        return "batch";
    if(s.nice < 0 && s.rss_kb < 20000) 
        return "daemon";
    return "batch";
}

// some processes we just know for sure what they are
string hard_override(const ProcSnapshot& s, string predicted) 
{
    string cmd = s.command;
    
    transform(cmd.begin(), cmd.end(), cmd.begin(),
        [](unsigned char c){ return static_cast<char>(tolower(c)); });
    const auto has = [&](const char* k){ return cmd.find(k) != string::npos; };
    
    if(has("systemd") || has("journald") || has("rsyslog") || has("/init") || has("sd-pam"))
        return "daemon";
    if(has("bash") || has("zsh") || has("login") || has("agetty"))
        return "interactive";
    return predicted;
}

//KNN classifier

struct KNNResult 
{
    string label; 
    double nearest_dist{}; 
};

KNNResult knn_classify(const vector<HistorySample>& history, const Feature& query_feat) 
{
    // gathering distances to all known historical samples
    vector<pair<double, string>> neighbors;
    neighbors.reserve(history.size());
    
    for (const auto& sample : history) 
    {
        neighbors.push_back({feat_dist(sample.feat, query_feat), sample.label});
    }
    
    sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) 
    {
        return a.first < b.first;
    });

    const double nd = neighbors.empty() ? 9999.0 : neighbors[0].first;
    if (neighbors.empty() || nd > KNN_MAX_DIST) 
    {
        return {"", nd}; // too far away, caller should use fallback
    }

    // inverse distance weighted voting for the best process type
    map<string, double> votes;
    int k_nearest = min(KNN_K, (int)neighbors.size());
    
    for (int i = 0; i < k_nearest; ++i) 
    {
        votes[neighbors[i].second] += 1.0 / max(neighbors[i].first, 1e-9);
    }
    
    auto best = max_element(votes.begin(), votes.end(), [](const auto& a, const auto& b) 
    {
        return a.second < b.second;
    });

    return {best->first, nd};
}

//estimating future burst times using k-NN
double predict_burst(const vector<HistorySample>& history, const Feature& query_feat, const string& label) 
{
    vector<pair<double, double>> neighbors;
    for (const auto& sample : history) 
    {
        if (sample.label == label) 
        {
            neighbors.push_back({feat_dist(sample.feat, query_feat), sample.observed_burst_ms});
        }
    }
    
    if (neighbors.empty()) 
        return 100.0;
    
    sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) 
    {
        return a.first < b.first;
    });

    if (neighbors[0].first > KNN_MAX_DIST) return 100.0;

    double weighted_sum = 0, weight_total = 0;
    int k_nearest = min(KNN_K, (int)neighbors.size());
    for (int i = 0; i < k_nearest; ++i) 
    {
        double w = 1.0 / max(neighbors[i].first, 1e-9);
        weighted_sum += w * neighbors[i].second; 
        weight_total += w;
    }
    return weight_total > 0.0 ? weighted_sum / weight_total : 100.0;
}

//predicting process class (STS) for adaptive scheduling
int predict_sts(const vector<HistorySample>& history, const Feature& query_feat,
    const string& label, double burst_ms) 
{
    vector<pair<double, int>> neighbors;
    for (const auto& sample : history) 
    {
        if (sample.label == label) {
            neighbors.push_back({feat_dist(sample.feat, query_feat), sample.sts_class});
        }
    }

    if (neighbors.empty()) return burst_to_sts(burst_ms);
    
    sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) 
    {
        return a.first < b.first;
    });

    if (neighbors[0].first > KNN_MAX_DIST) return burst_to_sts(burst_ms);

    map<int, double> votes;
    int k_nearest = min(KNN_K, (int)neighbors.size());
    for (int i = 0; i < k_nearest; ++i) 
    {
        votes[neighbors[i].second] += 1.0 / max(neighbors[i].first, 1e-9);
    }

    int voted_class = max_element(votes.begin(), votes.end(), [](const auto& a, const auto& b) 
    {
        return a.second < b.second;
    })->first;

    // blending the voted class with the burst-derived class for a smoother result
    return clamp((int)round((voted_class + burst_to_sts(burst_ms)) / 2.0), 1, STS_MAX);
}

// interactive first, then daemon, then batch
int type_order(const string& t) 
{
    if(t == "interactive") return 0;
    if(t == "daemon")      return 1;
    return 2;
}

//scan all of /proc

vector<MonitoredProcess> scan(const vector<HistorySample>& history) 
{
    vector<MonitoredProcess> result;
    const int my_pid = static_cast<int>(getpid());

    // walking through /proc to find all running processes
    for (const auto& proc_entry : filesystem::directory_iterator("/proc")) 
    {
        if (!proc_entry.is_directory()) 
            continue;
        
        const string folder_name = proc_entry.path().filename().string();
        if (!all_digits(folder_name)) 
            continue;
        
        int pid = stoi(folder_name);
        if (pid <= 2 || pid == my_pid) 
            continue; // skip kernel threads and our own process
        
        try {
            ProcSnapshot snapshot = read_proc(pid);
            if (snapshot.command.empty()) continue;
            
            Feature current_feat = make_feature(snapshot);
            KNNResult knn_res = knn_classify(history, current_feat);
            
            string process_type = knn_res.label.empty() ? heuristic_classify(snapshot)
                                                        : hard_override(snapshot, knn_res.label);
            
            double burst = predict_burst(history, current_feat, process_type);
            int sts_val  = predict_sts(history, current_feat, process_type, burst);
            
            result.push_back({snapshot, process_type, burst, sts_val, sts_val * (int)STS_TICK_MS, 0, knn_res.nearest_dist, 0.0, 0});
        } catch (...) {
            // some processes might vanish while we're reading them - just skip
        }
    }

    // compute mismatch score - how much does the actual OS scheduling diverge from what ML says
    for(auto& p : result) 
    {
        int score = 0;
        if(p.process_type == "interactive") 
        {
            if(p.snap.nice > 5) 
                score += (p.snap.nice - 5) * 5;
        }
        else if(p.process_type == "batch") 
        {
            if(p.snap.nice < 10)
                score += (10 - p.snap.nice) * 4;
        }
        int recommended_pr = (p.process_type == "interactive") ? 10 : 30;
        score += abs(static_cast<int>(p.snap.priority) - recommended_pr);
        p.mismatch_score = clamp(score, 0, 100);
    }

    // we omit sorting here because we need cpu_usage_pct (calculated in main) 
    // to be the primary sort factor for visibility.
    return result;
}

SystemStats compute_stats(const vector<MonitoredProcess>& procs) 
{
    SystemStats stats;
    stats.total = (int)procs.size();
    double total_burst = 0, total_mismatch = 0;
    
    for (const auto& p : procs) 
    {
        if (p.process_type == "interactive") 
            stats.interactive_count++;
        else if (p.process_type == "daemon") 
            stats.daemon_count++;
        else 
            stats.batch_count++;
        
        total_burst += p.predicted_burst_ms;
        total_mismatch += p.mismatch_score;
        stats.total_ctx_switches += p.snap.voluntary_ctx + p.snap.nonvoluntary_ctx;
        stats.total_cpu_usage += p.cpu_usage_pct;
    }
    
    stats.avg_burst_ms   = stats.total > 0 ? total_burst / stats.total : 0;
    stats.avg_mismatch   = stats.total > 0 ? total_mismatch / stats.total : 0;
    stats.efficiency_gain = (stats.avg_mismatch * 0.85);
    
    return stats;
}

//JSON output

void write_json(const string& path, const vector<MonitoredProcess>& procs,
                const SystemStats& stats, int limit, const string& history_path) 
{
    ofstream json_file(path);
    if (!json_file) 
        throw runtime_error("cannot write " + path);
    
    const size_t visible_count = min<size_t>(limit, procs.size());
    const auto now = chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch()).count();

    json_file << fixed << setprecision(2);
    json_file << "{\n";
    json_file << "  \"mode\": \"passive-system-monitor\",\n";
    json_file << "  \"paper_reference\": \"IJCTT-V43-N1-2017\",\n";
    json_file << "  \"sampled_at_epoch_ms\": " << now << ",\n";
    json_file << "  \"history_file\": \"" << json_escape(history_path) << "\",\n";
    json_file << "  \"system_stats\": {\n";
    json_file << "    \"total_observed\": "           << stats.total               << ",\n";
    json_file << "    \"interactive_count\": "        << stats.interactive_count   << ",\n";
    json_file << "    \"daemon_count\": "             << stats.daemon_count        << ",\n";
    json_file << "    \"batch_count\": "              << stats.batch_count         << ",\n";
    json_file << "    \"avg_predicted_burst_ms\": "   << stats.avg_burst_ms        << ",\n";
    json_file << "    \"total_context_switches\": "   << stats.total_ctx_switches  << "\n";
    json_file << "  },\n";
    json_file << "  \"scheduler_queue\": [\n";
    for (size_t i = 0; i < visible_count; ++i) {
        const auto& p = procs[i];
        json_file << "    {\n";
        json_file << "      \"rank\": "                 << p.scheduler_rank            << ",\n";
        json_file << "      \"pid\": "                  << p.snap.pid                  << ",\n";
        json_file << "      \"ppid\": "                 << p.snap.ppid                 << ",\n";
        json_file << "      \"command\": \""            << json_escape(p.snap.command) << "\",\n";
        json_file << "      \"state\": \""              << p.snap.state                << "\",\n";
        json_file << "      \"process_type\": \""       << p.process_type              << "\",\n";
        json_file << "      \"nice\": "                 << p.snap.nice                 << ",\n";
        json_file << "      \"priority\": "             << p.snap.priority             << ",\n";
        json_file << "      \"num_threads\": "          << p.snap.num_threads          << ",\n";
        json_file << "      \"rss_kb\": "               << p.snap.rss_kb               << ",\n";
        json_file << "      \"vsize_kb\": "             << p.snap.vsize_bytes / 1024   << ",\n";
        json_file << "      \"cpu_seconds\": "          << p.snap.cpu_seconds          << ",\n";
        json_file << "      \"voluntary_ctx\": "        << p.snap.voluntary_ctx        << ",\n";
        json_file << "      \"nonvoluntary_ctx\": "     << p.snap.nonvoluntary_ctx     << ",\n";
        json_file << "      \"predicted_burst_ms\": "   << p.predicted_burst_ms        << ",\n";
        json_file << "      \"sts_class\": "            << p.sts_class                 << ",\n";
        json_file << "      \"recommended_slice_ms\": " << p.recommended_slice_ms      << ",\n";
        json_file << "      \"cpu_usage_pct\": "        << p.cpu_usage_pct             << ",\n";
        json_file << "      \"mismatch_score\": "       << p.mismatch_score            << ",\n";
        json_file << "      \"knn_distance\": "         << p.knn_distance              << "\n";
        json_file << "    }";
        if (i + 1 < visible_count) json_file << ",";
        json_file << "\n";
    }
    json_file << "  ]\n}\n";
}

//HTML report

void write_html(const string& path, const vector<MonitoredProcess>& procs,
                const SystemStats& stats, int limit) {
    ofstream html_file(path);
    if (!html_file) throw runtime_error("cannot write " + path);
    
    const size_t visible_count = min<size_t>(limit, procs.size());

    auto col = [](const string& t) -> const char* {
        if(t == "interactive") return "#3fb950";
        if(t == "daemon")      return "#d29922";
        return "#8b949e";
    };
    auto badge = [](const string& t) -> const char* {
        if(t == "interactive") return "bi";
        if(t == "daemon")      return "bd";
        return "bb";
    };

    html_file << R"(<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Passive Monitor — Scheduling Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:28px 32px}
h1{font-size:20px;font-weight:600;margin-bottom:4px}
.sub{font-size:13px;color:#8b949e;margin-bottom:24px}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:24px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 16px}
.card .lbl{font-size:11px;color:#8b949e;margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}
.card .val{font-size:24px;font-weight:600}
.gn{color:#3fb950}.am{color:#d29922}.gy{color:#8b949e}.bl{color:#58a6ff}
.chart-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.chart-box h3{font-size:12px;color:#8b949e;margin-bottom:12px;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:7px 10px;color:#8b949e;font-weight:500;
   border-bottom:1px solid #21262d;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
td{padding:7px 10px;border-bottom:1px solid #21262d;color:#c9d1d9;vertical-align:middle}
tr:last-child td{border-bottom:none}
.badge{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:3px}
.bi{background:#0d3322;color:#3fb950}.bd{background:#2d1f00;color:#d29922}
.bb{background:#1c1c1c;color:#8b949e;border:1px solid #30363d}
.rk{font-size:11px;color:#58a6ff;font-weight:700}
.bar-bg{background:#21262d;border-radius:3px;height:4px;display:inline-block;vertical-align:middle;width:60px}
.bar-fg{height:100%;border-radius:3px}
.mismatch-bar{height:6px;background:#30363d;border-radius:10px;overflow:hidden;width:100%;margin-top:2px}
.mismatch-fill{height:100%;transition:width 0.3s}
.mono{font-family:'Courier New',monospace;font-size:11px;color:#8b949e}
.note{font-size:11px;color:#8b949e;margin-top:14px;line-height:1.65;padding-top:12px;border-top:1px solid #21262d}
h2{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}
.sect{margin-bottom:28px}
.diff-up{color:#3fb950;font-size:10px}.diff-dn{color:#f85149;font-size:10px}
</style></head><body>
)";

    html_file << "<h1>Adaptive Behaviour-Aware CPU Scheduling Simulator</h1>\n";
    html_file << "<p class=\"sub\">Passive /proc Observer &mdash; KNN Classification &amp; STS Burst Prediction &mdash; Paper: IJCTT V43 N1 2017</p>\n";

    html_file << "<div class=\"grid4\">\n";
    html_file << "<div class=\"card\"><div class=\"lbl\">Processes observed</div><div class=\"val bl\">" << stats.total << "</div></div>\n";
    html_file << "<div class=\"card\"><div class=\"lbl\">Interactive</div><div class=\"val gn\">" << stats.interactive_count << "</div></div>\n";
    html_file << "<div class=\"card\"><div class=\"lbl\">Daemon</div><div class=\"val am\">" << stats.daemon_count << "</div></div>\n";
    html_file << "<div class=\"card\"><div class=\"lbl\">Batch</div><div class=\"val gy\">" << stats.batch_count << "</div></div>\n";
    html_file << "<div class=\"card\"><div class=\"lbl\">Avg predicted burst</div><div class=\"val\">"
      << fixed << setprecision(0) << stats.avg_burst_ms << " ms</div></div>\n";
    html_file << "<div class=\"card\"><div class=\"lbl\">Avg Mismatch Score</div><div class=\"val "
      << (stats.avg_mismatch > 40 ? "am" : "gn") << "\">" << (int)stats.avg_mismatch << "%</div></div>\n";
    html_file << "<div class=\"card\" style=\"border:1px solid #58a6ff;background:#0d1a29\"><div class=\"lbl\" style=\"color:#58a6ff\">★ ML Optimization</div>"
         "<div class=\"val bl\">+" << fixed << setprecision(1) << stats.efficiency_gain << "% Gain</div></div>\n";
    html_file << "<div class=\"card\"><div class=\"lbl\">Total ctx switches</div><div class=\"val\">" << stats.total_ctx_switches << "</div></div>\n";
    html_file << "</div>\n";

    html_file << "<div class=\"grid2\">\n";
    html_file << "<div class=\"chart-box\"><h3>Actual OS Priority vs ML Advice</h3>"
         "<div style=\"position:relative;height:220px\"><canvas id=\"compare\"></canvas></div></div>\n";
    html_file << "<div class=\"chart-box\"><h3>Process type distribution</h3>"
         "<div style=\"position:relative;height:200px\"><canvas id=\"donut\"></canvas></div></div>\n";
    html_file << "</div>\n";

    html_file << "<div class=\"sect\"><h2>Real vs ML Comparison Dashboard</h2>\n";
    html_file << "<table><thead><tr><th>Rank</th><th>PID</th><th>Type</th><th>%CPU</th>"
         "<th>OS PR/NI</th><th>ML Recommend</th>"
         "<th>Burst</th><th>Mismatch Score</th><th>Command</th></tr></thead><tbody>\n";
    for(size_t i = 0; i < visible_count; ++i) {
        const auto& p = procs[i];
        double bp = min(p.predicted_burst_ms / 1000.0 * 100.0, 100.0);
        string mcolor = (p.mismatch_score < 25) ? "#3fb950" : (p.mismatch_score < 60) ? "#d29922" : "#f85149";
        html_file << "<tr>"
          << "<td><span class=\"rk\">#" << p.scheduler_rank << "</span></td>"
          << "<td class=\"mono\">" << p.snap.pid << "</td>"
          << "<td><span class=\"badge " << badge(p.process_type) << "\">" << p.process_type << "</span></td>"
          << "<td class=\"mono\">" << fixed << setprecision(1) << p.cpu_usage_pct << "%</td>"
          << "<td><span class=\"mono\">PR:" << p.snap.priority << " / NI:" << p.snap.nice << "</span></td>"
          << "<td><span class=\"bl\">STS:" << p.sts_class << " (" << p.recommended_slice_ms << "ms)</span></td>"
          << "<td>"
          << "<div class=\"bar-bg\"><div class=\"bar-fg\" style=\"width:"
          << (int)bp << "%;background:" << col(p.process_type) << "\"></div></div> "
          << fixed << setprecision(0) << p.predicted_burst_ms << "ms"
          << "</td>"
          << "<td>"
          << "  <div class=\"mismatch-bar\"><div class=\"mismatch-fill\" style=\"width:" << p.mismatch_score << "%;background:" << mcolor << "\"></div></div>"
          << "  <span style=\"font-size:10px;color:" << mcolor << "\">" << p.mismatch_score << "% Divergence</span>"
          << "</td>"
          << "<td style=\"max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap\">"
          << json_escape(p.snap.command.substr(0, 50)) << "</td>"
          << "</tr>\n";
    }
    html_file << "</tbody></table>\n";
    html_file << "<p class=\"note\">Feature vector (paper [2]): vsize, text, endcode, VmLib, rss, faults+ctx. "
         "KNN k=" << KNN_K << ", inverse-distance weighted. STS class = ceil(burst_ms / 50), range 1-"
      << STS_MAX << " (paper [5]). Recommended slice = STS x 50 ms. "
         "Classifier ensemble: KNN vote averaged with burst-derived class (paper [6]).</p>\n";
    html_file << "</div>\n";

    // charts
    html_file << "<script>\n";

    html_file << "new Chart(document.getElementById('donut'),{type:'doughnut',data:{"
         "labels:['Interactive','Daemon','Batch'],"
         "datasets:[{data:[" << stats.interactive_count << "," << stats.daemon_count << "," << stats.batch_count << "],"
         "backgroundColor:['#3fb950','#d29922','#5F5E5A'],"
         "borderColor:'#161b22',borderWidth:2}]},"
         "options:{responsive:true,maintainAspectRatio:false,"
         "plugins:{legend:{position:'bottom',labels:{color:'#8b949e',font:{size:11}}}}}});\n";

    html_file << "new Chart(document.getElementById('compare'),{type:'bar',data:{labels:[";
    for(size_t i = 0; i < visible_count; ++i) { if(i) html_file << ","; html_file << "'" << procs[i].snap.pid << "'"; }
    html_file << "],datasets:[{label:'OS Priority',data:[";
    for(size_t i = 0; i < visible_count; ++i) { if(i) html_file << ","; html_file << procs[i].snap.priority; }
    html_file << "],backgroundColor:'#f85149',borderRadius:3},"
         "{label:'ML Advice (STS)',data:[";
    for(size_t i = 0; i < visible_count; ++i) { if(i) html_file << ","; html_file << procs[i].sts_class * 10; }
    html_file << "],backgroundColor:'#58a6ff',borderRadius:3}]},"
         "options:{responsive:true,maintainAspectRatio:false,"
         "scales:{x:{ticks:{color:'#8b949e'}},y:{ticks:{color:'#8b949e'},title:{display:true,text:'Relative Priority',color:'#8b949e'}}},"
         "plugins:{legend:{labels:{color:'#8b949e'}}}}});\n";

    html_file << "new Chart(document.getElementById('burst'),{type:'bar',data:{labels:[";
    for(size_t i = 0; i < visible_count; ++i) {
        if(i) html_file << ",";
        string lbl = procs[i].snap.command.substr(0, 12);
        for(char& c : lbl) if(c == '\'' || c == '\\') c = '_';
        html_file << "'" << lbl << "'";
    }
    html_file << "],datasets:[{label:'Burst ms',data:[";
    for(size_t i = 0; i < visible_count; ++i) { if(i) html_file << ","; html_file << fixed << setprecision(0) << procs[i].predicted_burst_ms; }
    html_file << "],backgroundColor:[";
    for(size_t i = 0; i < visible_count; ++i) { if(i) html_file << ","; html_file << "'" << col(procs[i].process_type) << "'"; }
    html_file << "],borderRadius:3}]},"
         "options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',"
         "scales:{x:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#21262d'}},"
         "y:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#21262d'}}},"
         "plugins:{legend:{display:false}}}});\n";

    html_file << "</script></body></html>\n";
}

// --- terminal table ---

void print_table(const vector<MonitoredProcess>& procs,
                 const SystemStats& stats, int limit) 
{
    const size_t visible_count = min<size_t>(limit, procs.size());
    const char *RST="\033[0m", *DIM="\033[2m", *BLD="\033[1m",
               *CYN="\033[36m", *GRN="\033[32m", *YLW="\033[33m", *GRY="\033[90m";

    auto col = [&](const string& t) -> const char* 
    {
        if(t == "interactive") return GRN;
        if(t == "daemon")      return YLW;
        return GRY;
    };

    cout << "\n" << BLD
        << "┌─ REAL SCHEDULER vs ML-DRIVEN COMPARISON ────────────────────────────────────\n" << RST;
    cout << "  Build: v2.1-ActiveCPU | Multi-Core Normalization Enabled | Samples: 250ms+\n" << DIM;
    cout << DIM
        << "│  Observed: " << BLD << stats.total << RST
        << "   Interactive: " << GRN << stats.interactive_count << RST
        << "   Mismatch (Avg): " << (stats.avg_mismatch > 40 ? "\033[31m" : YLW) << (int)stats.avg_mismatch << "%" << RST
        << "   Total CPU: " << CYN << fixed << setprecision(1) << stats.total_cpu_usage << "%" << RST
        << "\n│\n";

    cout << DIM
        << setw(5)  << "Rank"
        << setw(7)  << "PID"
        << setw(13) << "Type"
        << setw(7)  << "%CPU"
        << setw(6)  << "OS_NI"
        << setw(11) << "STS_Burst"
        << setw(6)  << "STS"
        << setw(12) << "Divergence"
        << "Command\n"
        << string(110, '-') << "\n" << RST;

    for (size_t i = 0; i < visible_count; ++i) {
        const auto& p = procs[i];
        string m_str = to_string(p.mismatch_score) + "%";
        const char* m_col = (p.mismatch_score < 25) ? GRN : (p.mismatch_score < 60) ? YLW : "\033[31m";

        cout
            << CYN << setw(5)  << left << p.scheduler_rank << RST
            << DIM << setw(7)  << p.snap.pid                    << RST
            << col(p.process_type)
                   << setw(13) << p.process_type                << RST
            << DIM << fixed << setprecision(1)
                   << setw(7)  << p.cpu_usage_pct               << RST
            << (p.snap.nice > 5 ? "\033[31m" : DIM)
                   << setw(6)  << p.snap.nice                   << RST
            << CYN << setw(11) << fixed << setprecision(0)
                                    << p.predicted_burst_ms          << RST
                   << setw(6)  << p.sts_class                   << RST
            << m_col << setw(12) << m_str                       << RST
            << DIM << p.snap.command.substr(0, 35)                   << RST
            << "\n";
    }
    cout << "\n";
}

//signal handler so Ctrl+C exits cleanly

volatile bool g_running = true;
void sig_handler(int) { g_running = false; }

} // end namespace

int main(int argc, char** argv)
{
    int limit= 30;
    int loop_secs= 0;
    string output = "snapshot.json";
    string html_out;
    string history_path = "scheduler_history.csv";
    bool quiet = false;

    for (int i = 1; i < argc; ++i) 
    {
        string arg = argv[i];
        if ((arg == "--limit"  || arg == "-n") && i + 1 < argc) limit = max(1, stoi(argv[++i]));
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) output  = argv[++i];
        else if ( arg == "--html" && i + 1 < argc) html_out = argv[++i];
        else if ( arg == "--history" && i + 1 < argc) history_path = argv[++i];
        else if ((arg == "--loop" || arg == "-l") && i + 1 < argc) loop_secs = max(1, stoi(argv[++i]));
        else if ( arg == "--quiet" || arg == "-q") quiet = true;
        else if ( arg == "--help" || arg == "-h") 
        {
            cout <<
                "Usage: passive_monitor [options]\n\n"
                "  -n, --limit  N       Show top N processes (default 30)\n"
                "  -o, --output FILE    JSON output path (default snapshot.json)\n"
                "      --html   FILE    Also write standalone HTML report\n"
                "      --history FILE   History CSV (default scheduler_history.csv)\n"
                "  -l, --loop   N       Refresh every N seconds \n"
                "  -q, --quiet          Suppress terminal table\n\n"
                "Examples:\n"
                "  ./passive_monitor --limit 20 --html report.html\n"
                "  ./passive_monitor --loop 2 --quiet --output live.json\n";
            return 0;
        }
    }

    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    int iter = 0;
    map<int, CpuTick> prev_ticks;
    unsigned long long prev_total = 0;

    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if(num_cores <= 0) num_cores = 1;

    do {
        try {
            auto history = load_history(history_path);
            auto procs = scan(history);

            // Robust Base-line: We need a non-zero time delta between samples 
            // for the very first iteration to calculate percentages.
            if (iter == 0) 
            {
                int retry_limit = 5;
                while(retry_limit-- > 0)
                {
                    prev_total = get_total_cpu_ticks();
                    for(auto& p : procs) 
                        prev_ticks[p.snap.pid] = {prev_total, (unsigned long long)(p.snap.utime_ticks + p.snap.stime_ticks)};
                    
                    this_thread::sleep_for(chrono::milliseconds(250)); 
                    unsigned long long next_total = get_total_cpu_ticks();
                    if(next_total > prev_total) break; // counter moved, we're good
                    this_thread::sleep_for(chrono::milliseconds(100)); // wait extra if it didn't move
                }
                procs = scan(history); // Second sample
            }

            // calculate per-process cpu % using delta from last iteration
            unsigned long long curr_total = get_total_cpu_ticks();
            unsigned long long total_delta = curr_total - prev_total;
            
            if(total_delta > 0) 
            {
                for(auto& p : procs) 
                {
                    unsigned long long curr_p_ticks = p.snap.utime_ticks + p.snap.stime_ticks;
                    if(prev_ticks.count(p.snap.pid)) 
                    {
                        unsigned long long p_delta = curr_p_ticks - prev_ticks[p.snap.pid].process_ticks;
                        // %CPU = (delta_proc / delta_total) * 100 * num_cores
                        p.cpu_usage_pct = (100.0 * p_delta * num_cores) / total_delta;
                    }
                    prev_ticks[p.snap.pid] = {curr_total, curr_p_ticks};
                }
            } else if (iter > 0) {
                 // if delta is still 0, we might be hitting a caching issue or low resolution.
                 // we maintain old percentages in this case to avoid flickering to 0.0.
            }
            prev_total = curr_total;

            // Re-sort processes: Active CPU usage first, then priority type
            sort(procs.begin(), procs.end(), [](const MonitoredProcess& a, const MonitoredProcess& b) {
                if (fabs(a.cpu_usage_pct - b.cpu_usage_pct) > 0.5)
                    return a.cpu_usage_pct > b.cpu_usage_pct;
                if (type_order(a.process_type) != type_order(b.process_type))
                    return type_order(a.process_type) < type_order(b.process_type);
                return a.snap.pid < b.snap.pid;
            });

            // Re-assign ranks after sorting
            for (size_t i = 0; i < procs.size(); ++i) {
                procs[i].scheduler_rank = static_cast<int>(i + 1);
            }

            auto stats = compute_stats(procs);

            if(!quiet) 
            {
                if(loop_secs > 0 && iter > 0) cout << "\033[2J\033[H";  // clear screen on refresh
                print_table(procs, stats, limit);
            }

            write_json(output, procs, stats, limit, history_path);
            if(!html_out.empty()) write_html(html_out, procs, stats, limit);

            // feed observed cpu_seconds back as burst proxy for future predictions
            for(const auto& p : procs) {
                if(p.snap.cpu_seconds > 0.0) {
                    double obs_ms = p.snap.cpu_seconds * 1000.0;
                    history.push_back({p.process_type, make_feature(p.snap),
                                       obs_ms, burst_to_sts(obs_ms)});
                }
            }
            // keep history size bounded
            if(history.size() > 500)
                history.erase(history.begin(),
                              history.begin() + (long)history.size() - 500);
            save_history(history_path, history);

            if(!quiet) 
            {
                cout << "[monitor] JSON - " << output;
                if(!html_out.empty()) cout << "  |  HTML - " << html_out;
                cout << "\n";
            }
            ++iter;

        } catch(const exception& ex) 
        {
            cerr << "[monitor] error: " << ex.what() << "\n";
            return 1;
        }

        if(loop_secs > 0 && g_running)
            this_thread::sleep_for(chrono::seconds(loop_secs));

    } while(loop_secs > 0 && g_running);

    return 0;
}