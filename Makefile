# ── Adaptive Behaviour-Aware CPU Scheduling Simulator ──────────────────────
# Makefile

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
TARGET   := passive_monitor
SRC      := passive_monitor.cpp

.PHONY: all clean run demo

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo ""
	@echo "  Build successful: ./$(TARGET)"
	@echo ""
	@echo "  Quick-start:"
	@echo "    ./$(TARGET)                          # single snapshot, limit 15"
	@echo "    ./$(TARGET) --limit 20               # show top 20"
	@echo "    ./$(TARGET) --html report.html       # write HTML report"
	@echo "    ./$(TARGET) --loop 3                 # live refresh every 3 s"
	@echo "    ./$(TARGET) --help                   # all options"
	@echo ""

# Run a single snapshot and open HTML report
demo: $(TARGET)
	./$(TARGET) --limit 20 --output snapshot.json --html report.html
	@echo ""
	@echo "  Outputs: snapshot.json  report.html  scheduler_history.csv"

clean:
	rm -f $(TARGET) snapshot.json report.html scheduler_history.csv
