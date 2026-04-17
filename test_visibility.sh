#!/bin/bash
echo "int main(){ volatile int i=0; while(1) i++; return 0; }" > loop.c
gcc -O0 loop.c -o loop_test
./loop_test &
LOOP_PID=$!
sleep 2
./monitor --limit 30 --html report_test.html --output snapshot_test.json
kill $LOOP_PID
rm loop.c loop_test
