#!/bin/bash

# Enable Python fault handler to catch segfault lines
export PYTHONFAULTHANDLER=1

# Create a log directory
LOG_DIR="/home/camerop/AC/logs"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Processing: aeronetgalata_2025-01-02T08-52-34Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetgalata_2025-01-02T08-52-34Z > "$LOG_DIR/aeronetgalata_2025-01-02T08-52-34Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetgalata_2025-01-02T08-52-34Z crashed (Check log for details)"
    echo "aeronetgalata_2025-01-02T08-52-34Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetgalata_2025-01-02T08-52-34Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-03-04T10-38-05Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-03-04T10-38-05Z > "$LOG_DIR/aeronetvenice_2025-03-04T10-38-05Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-03-04T10-38-05Z crashed (Check log for details)"
    echo "aeronetvenice_2025-03-04T10-38-05Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-03-04T10-38-05Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-05-14T10-45-06Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-05-14T10-45-06Z > "$LOG_DIR/aeronetvenice_2025-05-14T10-45-06Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-05-14T10-45-06Z crashed (Check log for details)"
    echo "aeronetvenice_2025-05-14T10-45-06Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-05-14T10-45-06Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-06-12T09-58-02Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-06-12T09-58-02Z > "$LOG_DIR/aeronetvenice_2025-06-12T09-58-02Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-06-12T09-58-02Z crashed (Check log for details)"
    echo "aeronetvenice_2025-06-12T09-58-02Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-06-12T09-58-02Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-06-22T10-46-15Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-06-22T10-46-15Z > "$LOG_DIR/aeronetvenice_2025-06-22T10-46-15Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-06-22T10-46-15Z crashed (Check log for details)"
    echo "aeronetvenice_2025-06-22T10-46-15Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-06-22T10-46-15Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-07-22T09-57-52Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-07-22T09-57-52Z > "$LOG_DIR/aeronetvenice_2025-07-22T09-57-52Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-07-22T09-57-52Z crashed (Check log for details)"
    echo "aeronetvenice_2025-07-22T09-57-52Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-07-22T09-57-52Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-07-23T10-02-32Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-07-23T10-02-32Z > "$LOG_DIR/aeronetvenice_2025-07-23T10-02-32Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-07-23T10-02-32Z crashed (Check log for details)"
    echo "aeronetvenice_2025-07-23T10-02-32Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-07-23T10-02-32Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-09-04T10-06-53Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-09-04T10-06-53Z > "$LOG_DIR/aeronetvenice_2025-09-04T10-06-53Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-09-04T10-06-53Z crashed (Check log for details)"
    echo "aeronetvenice_2025-09-04T10-06-53Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-09-04T10-06-53Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2025-09-25T10-01-52Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-09-25T10-01-52Z > "$LOG_DIR/aeronetvenice_2025-09-25T10-01-52Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2025-09-25T10-01-52Z crashed (Check log for details)"
    echo "aeronetvenice_2025-09-25T10-01-52Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2025-09-25T10-01-52Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2026-01-10T10-04-18Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2026-01-10T10-04-18Z > "$LOG_DIR/aeronetvenice_2026-01-10T10-04-18Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2026-01-10T10-04-18Z crashed (Check log for details)"
    echo "aeronetvenice_2026-01-10T10-04-18Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2026-01-10T10-04-18Z completed."
fi

echo "========================================="
echo "Processing: aeronetvenice_2026-06-19T10-16-46Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/aeronetvenice_2026-06-19T10-16-46Z > "$LOG_DIR/aeronetvenice_2026-06-19T10-16-46Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: aeronetvenice_2026-06-19T10-16-46Z crashed (Check log for details)"
    echo "aeronetvenice_2026-06-19T10-16-46Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: aeronetvenice_2026-06-19T10-16-46Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-02-04T15-46-53Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-02-04T15-46-53Z > "$LOG_DIR/annapolis_2025-02-04T15-46-53Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-02-04T15-46-53Z crashed (Check log for details)"
    echo "annapolis_2025-02-04T15-46-53Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-02-04T15-46-53Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-05-17T15-51-35Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-05-17T15-51-35Z > "$LOG_DIR/annapolis_2025-05-17T15-51-35Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-05-17T15-51-35Z crashed (Check log for details)"
    echo "annapolis_2025-05-17T15-51-35Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-05-17T15-51-35Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-07-22T16-24-24Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-07-22T16-24-24Z > "$LOG_DIR/annapolis_2025-07-22T16-24-24Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-07-22T16-24-24Z crashed (Check log for details)"
    echo "annapolis_2025-07-22T16-24-24Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-07-22T16-24-24Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-08-25T15-49-07Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-08-25T15-49-07Z > "$LOG_DIR/annapolis_2025-08-25T15-49-07Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-08-25T15-49-07Z crashed (Check log for details)"
    echo "annapolis_2025-08-25T15-49-07Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-08-25T15-49-07Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-08-26T15-53-35Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-08-26T15-53-35Z > "$LOG_DIR/annapolis_2025-08-26T15-53-35Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-08-26T15-53-35Z crashed (Check log for details)"
    echo "annapolis_2025-08-26T15-53-35Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-08-26T15-53-35Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-10-16T16-18-10Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-10-16T16-18-10Z > "$LOG_DIR/annapolis_2025-10-16T16-18-10Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-10-16T16-18-10Z crashed (Check log for details)"
    echo "annapolis_2025-10-16T16-18-10Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-10-16T16-18-10Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-11-05T15-59-32Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-11-05T15-59-32Z > "$LOG_DIR/annapolis_2025-11-05T15-59-32Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-11-05T15-59-32Z crashed (Check log for details)"
    echo "annapolis_2025-11-05T15-59-32Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-11-05T15-59-32Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-11-08T16-10-39Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-11-08T16-10-39Z > "$LOG_DIR/annapolis_2025-11-08T16-10-39Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-11-08T16-10-39Z crashed (Check log for details)"
    echo "annapolis_2025-11-08T16-10-39Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-11-08T16-10-39Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-11-29T15-50-03Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-11-29T15-50-03Z > "$LOG_DIR/annapolis_2025-11-29T15-50-03Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-11-29T15-50-03Z crashed (Check log for details)"
    echo "annapolis_2025-11-29T15-50-03Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-11-29T15-50-03Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2025-12-01T15-56-56Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2025-12-01T15-56-56Z > "$LOG_DIR/annapolis_2025-12-01T15-56-56Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2025-12-01T15-56-56Z crashed (Check log for details)"
    echo "annapolis_2025-12-01T15-56-56Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2025-12-01T15-56-56Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2026-03-10T16-03-45Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2026-03-10T16-03-45Z > "$LOG_DIR/annapolis_2026-03-10T16-03-45Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2026-03-10T16-03-45Z crashed (Check log for details)"
    echo "annapolis_2026-03-10T16-03-45Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2026-03-10T16-03-45Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2026-04-11T15-46-47Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2026-04-11T15-46-47Z > "$LOG_DIR/annapolis_2026-04-11T15-46-47Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2026-04-11T15-46-47Z crashed (Check log for details)"
    echo "annapolis_2026-04-11T15-46-47Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2026-04-11T15-46-47Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2026-04-12T15-49-06Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2026-04-12T15-49-06Z > "$LOG_DIR/annapolis_2026-04-12T15-49-06Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2026-04-12T15-49-06Z crashed (Check log for details)"
    echo "annapolis_2026-04-12T15-49-06Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2026-04-12T15-49-06Z completed."
fi

echo "========================================="
echo "Processing: annapolis_2026-05-31T16-03-20Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/annapolis_2026-05-31T16-03-20Z > "$LOG_DIR/annapolis_2026-05-31T16-03-20Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: annapolis_2026-05-31T16-03-20Z crashed (Check log for details)"
    echo "annapolis_2026-05-31T16-03-20Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: annapolis_2026-05-31T16-03-20Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-02-11T02-05-25Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-02-11T02-05-25Z > "$LOG_DIR/ariake_2025-02-11T02-05-25Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-02-11T02-05-25Z crashed (Check log for details)"
    echo "ariake_2025-02-11T02-05-25Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-02-11T02-05-25Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-04-04T02-27-25Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-04-04T02-27-25Z > "$LOG_DIR/ariake_2025-04-04T02-27-25Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-04-04T02-27-25Z crashed (Check log for details)"
    echo "ariake_2025-04-04T02-27-25Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-04-04T02-27-25Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-07-23T02-04-14Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-07-23T02-04-14Z > "$LOG_DIR/ariake_2025-07-23T02-04-14Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-07-23T02-04-14Z crashed (Check log for details)"
    echo "ariake_2025-07-23T02-04-14Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-07-23T02-04-14Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-08-15T02-14-38Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-08-15T02-14-38Z > "$LOG_DIR/ariake_2025-08-15T02-14-38Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-08-15T02-14-38Z crashed (Check log for details)"
    echo "ariake_2025-08-15T02-14-38Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-08-15T02-14-38Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-08-16T02-19-08Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-08-16T02-19-08Z > "$LOG_DIR/ariake_2025-08-16T02-19-08Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-08-16T02-19-08Z crashed (Check log for details)"
    echo "ariake_2025-08-16T02-19-08Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-08-16T02-19-08Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-09-30T02-24-36Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-09-30T02-24-36Z > "$LOG_DIR/ariake_2025-09-30T02-24-36Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-09-30T02-24-36Z crashed (Check log for details)"
    echo "ariake_2025-09-30T02-24-36Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-09-30T02-24-36Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-10-17T01-57-43Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-10-17T01-57-43Z > "$LOG_DIR/ariake_2025-10-17T01-57-43Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-10-17T01-57-43Z crashed (Check log for details)"
    echo "ariake_2025-10-17T01-57-43Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-10-17T01-57-43Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-11-15T02-11-53Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-11-15T02-11-53Z > "$LOG_DIR/ariake_2025-11-15T02-11-53Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-11-15T02-11-53Z crashed (Check log for details)"
    echo "ariake_2025-11-15T02-11-53Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-11-15T02-11-53Z completed."
fi

echo "========================================="
echo "Processing: ariake_2025-11-16T02-15-27Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2025-11-16T02-15-27Z > "$LOG_DIR/ariake_2025-11-16T02-15-27Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2025-11-16T02-15-27Z crashed (Check log for details)"
    echo "ariake_2025-11-16T02-15-27Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2025-11-16T02-15-27Z completed."
fi

echo "========================================="
echo "Processing: ariake_2026-01-14T02-18-38Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2026-01-14T02-18-38Z > "$LOG_DIR/ariake_2026-01-14T02-18-38Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2026-01-14T02-18-38Z crashed (Check log for details)"
    echo "ariake_2026-01-14T02-18-38Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2026-01-14T02-18-38Z completed."
fi

echo "========================================="
echo "Processing: ariake_2026-01-17T02-27-37Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2026-01-17T02-27-37Z > "$LOG_DIR/ariake_2026-01-17T02-27-37Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2026-01-17T02-27-37Z crashed (Check log for details)"
    echo "ariake_2026-01-17T02-27-37Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2026-01-17T02-27-37Z completed."
fi

echo "========================================="
echo "Processing: ariake_2026-01-18T02-30-37Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2026-01-18T02-30-37Z > "$LOG_DIR/ariake_2026-01-18T02-30-37Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2026-01-18T02-30-37Z crashed (Check log for details)"
    echo "ariake_2026-01-18T02-30-37Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2026-01-18T02-30-37Z completed."
fi

echo "========================================="
echo "Processing: ariake_2026-02-18T02-22-47Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2026-02-18T02-22-47Z > "$LOG_DIR/ariake_2026-02-18T02-22-47Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2026-02-18T02-22-47Z crashed (Check log for details)"
    echo "ariake_2026-02-18T02-22-47Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2026-02-18T02-22-47Z completed."
fi

echo "========================================="
echo "Processing: ariake_2026-05-09T02-27-11Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ariake_2026-05-09T02-27-11Z > "$LOG_DIR/ariake_2026-05-09T02-27-11Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ariake_2026-05-09T02-27-11Z crashed (Check log for details)"
    echo "ariake_2026-05-09T02-27-11Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ariake_2026-05-09T02-27-11Z completed."
fi

echo "========================================="
echo "Processing: chesapeake_2025-02-24T16-16-55Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/chesapeake_2025-02-24T16-16-55Z > "$LOG_DIR/chesapeake_2025-02-24T16-16-55Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: chesapeake_2025-02-24T16-16-55Z crashed (Check log for details)"
    echo "chesapeake_2025-02-24T16-16-55Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: chesapeake_2025-02-24T16-16-55Z completed."
fi

echo "========================================="
echo "Processing: cocobeach_2025-03-12T16-20-14Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/cocobeach_2025-03-12T16-20-14Z > "$LOG_DIR/cocobeach_2025-03-12T16-20-14Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: cocobeach_2025-03-12T16-20-14Z crashed (Check log for details)"
    echo "cocobeach_2025-03-12T16-20-14Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: cocobeach_2025-03-12T16-20-14Z completed."
fi

echo "========================================="
echo "Processing: cocobeach_2025-04-11T15-58-34Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/cocobeach_2025-04-11T15-58-34Z > "$LOG_DIR/cocobeach_2025-04-11T15-58-34Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: cocobeach_2025-04-11T15-58-34Z crashed (Check log for details)"
    echo "cocobeach_2025-04-11T15-58-34Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: cocobeach_2025-04-11T15-58-34Z completed."
fi

echo "========================================="
echo "Processing: cocobeach_2025-04-14T16-14-38Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/cocobeach_2025-04-14T16-14-38Z > "$LOG_DIR/cocobeach_2025-04-14T16-14-38Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: cocobeach_2025-04-14T16-14-38Z crashed (Check log for details)"
    echo "cocobeach_2025-04-14T16-14-38Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: cocobeach_2025-04-14T16-14-38Z completed."
fi

echo "========================================="
echo "Processing: cocobeach_2026-02-10T16-27-40Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/cocobeach_2026-02-10T16-27-40Z > "$LOG_DIR/cocobeach_2026-02-10T16-27-40Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: cocobeach_2026-02-10T16-27-40Z crashed (Check log for details)"
    echo "cocobeach_2026-02-10T16-27-40Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: cocobeach_2026-02-10T16-27-40Z completed."
fi

echo "========================================="
echo "Processing: frohavet_2025-02-25T11-26-39Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/frohavet_2025-02-25T11-26-39Z > "$LOG_DIR/frohavet_2025-02-25T11-26-39Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: frohavet_2025-02-25T11-26-39Z crashed (Check log for details)"
    echo "frohavet_2025-02-25T11-26-39Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: frohavet_2025-02-25T11-26-39Z completed."
fi

echo "========================================="
echo "Processing: image61N6E_2025-03-13T11-27-56Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/image61N6E_2025-03-13T11-27-56Z > "$LOG_DIR/image61N6E_2025-03-13T11-27-56Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: image61N6E_2025-03-13T11-27-56Z crashed (Check log for details)"
    echo "image61N6E_2025-03-13T11-27-56Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: image61N6E_2025-03-13T11-27-56Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-01-22T01-31-13Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-01-22T01-31-13Z > "$LOG_DIR/kemigawa_2025-01-22T01-31-13Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-01-22T01-31-13Z crashed (Check log for details)"
    echo "kemigawa_2025-01-22T01-31-13Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-01-22T01-31-13Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-02-05T01-26-20Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-02-05T01-26-20Z > "$LOG_DIR/kemigawa_2025-02-05T01-26-20Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-02-05T01-26-20Z crashed (Check log for details)"
    echo "kemigawa_2025-02-05T01-26-20Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-02-05T01-26-20Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-06-05T01-24-28Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-06-05T01-24-28Z > "$LOG_DIR/kemigawa_2025-06-05T01-24-28Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-06-05T01-24-28Z crashed (Check log for details)"
    echo "kemigawa_2025-06-05T01-24-28Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-06-05T01-24-28Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-08-04T01-23-24Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-08-04T01-23-24Z > "$LOG_DIR/kemigawa_2025-08-04T01-23-24Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-08-04T01-23-24Z crashed (Check log for details)"
    echo "kemigawa_2025-08-04T01-23-24Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-08-04T01-23-24Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-08-26T01-27-48Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-08-26T01-27-48Z > "$LOG_DIR/kemigawa_2025-08-26T01-27-48Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-08-26T01-27-48Z crashed (Check log for details)"
    echo "kemigawa_2025-08-26T01-27-48Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-08-26T01-27-48Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-08-27T01-32-16Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-08-27T01-32-16Z > "$LOG_DIR/kemigawa_2025-08-27T01-32-16Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-08-27T01-32-16Z crashed (Check log for details)"
    echo "kemigawa_2025-08-27T01-32-16Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-08-27T01-32-16Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-11-03T01-26-43Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-11-03T01-26-43Z > "$LOG_DIR/kemigawa_2025-11-03T01-26-43Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-11-03T01-26-43Z crashed (Check log for details)"
    echo "kemigawa_2025-11-03T01-26-43Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-11-03T01-26-43Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-11-28T01-21-24Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-11-28T01-21-24Z > "$LOG_DIR/kemigawa_2025-11-28T01-21-24Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-11-28T01-21-24Z crashed (Check log for details)"
    echo "kemigawa_2025-11-28T01-21-24Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-11-28T01-21-24Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-11-30T01-28-19Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-11-30T01-28-19Z > "$LOG_DIR/kemigawa_2025-11-30T01-28-19Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-11-30T01-28-19Z crashed (Check log for details)"
    echo "kemigawa_2025-11-30T01-28-19Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-11-30T01-28-19Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-12-01T01-31-45Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-12-01T01-31-45Z > "$LOG_DIR/kemigawa_2025-12-01T01-31-45Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-12-01T01-31-45Z crashed (Check log for details)"
    echo "kemigawa_2025-12-01T01-31-45Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-12-01T01-31-45Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-12-04T01-41-59Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-12-04T01-41-59Z > "$LOG_DIR/kemigawa_2025-12-04T01-41-59Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-12-04T01-41-59Z crashed (Check log for details)"
    echo "kemigawa_2025-12-04T01-41-59Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-12-04T01-41-59Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-12-07T01-52-02Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-12-07T01-52-02Z > "$LOG_DIR/kemigawa_2025-12-07T01-52-02Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-12-07T01-52-02Z crashed (Check log for details)"
    echo "kemigawa_2025-12-07T01-52-02Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-12-07T01-52-02Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-12-08T01-55-21Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-12-08T01-55-21Z > "$LOG_DIR/kemigawa_2025-12-08T01-55-21Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-12-08T01-55-21Z crashed (Check log for details)"
    echo "kemigawa_2025-12-08T01-55-21Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-12-08T01-55-21Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-12-28T01-25-19Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-12-28T01-25-19Z > "$LOG_DIR/kemigawa_2025-12-28T01-25-19Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-12-28T01-25-19Z crashed (Check log for details)"
    echo "kemigawa_2025-12-28T01-25-19Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-12-28T01-25-19Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2025-12-30T01-31-38Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2025-12-30T01-31-38Z > "$LOG_DIR/kemigawa_2025-12-30T01-31-38Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2025-12-30T01-31-38Z crashed (Check log for details)"
    echo "kemigawa_2025-12-30T01-31-38Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2025-12-30T01-31-38Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2026-01-27T01-20-45Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2026-01-27T01-20-45Z > "$LOG_DIR/kemigawa_2026-01-27T01-20-45Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2026-01-27T01-20-45Z crashed (Check log for details)"
    echo "kemigawa_2026-01-27T01-20-45Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2026-01-27T01-20-45Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2026-03-11T01-41-37Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2026-03-11T01-41-37Z > "$LOG_DIR/kemigawa_2026-03-11T01-41-37Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2026-03-11T01-41-37Z crashed (Check log for details)"
    echo "kemigawa_2026-03-11T01-41-37Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2026-03-11T01-41-37Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2026-03-15T01-51-41Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2026-03-15T01-51-41Z > "$LOG_DIR/kemigawa_2026-03-15T01-51-41Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2026-03-15T01-51-41Z crashed (Check log for details)"
    echo "kemigawa_2026-03-15T01-51-41Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2026-03-15T01-51-41Z completed."
fi

echo "========================================="
echo "Processing: kemigawa_2026-05-30T01-36-49Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/kemigawa_2026-05-30T01-36-49Z > "$LOG_DIR/kemigawa_2026-05-30T01-36-49Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: kemigawa_2026-05-30T01-36-49Z crashed (Check log for details)"
    echo "kemigawa_2026-05-30T01-36-49Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: kemigawa_2026-05-30T01-36-49Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-01-14T13-48-20Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-01-14T13-48-20Z > "$LOG_DIR/laplata_2025-01-14T13-48-20Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-01-14T13-48-20Z crashed (Check log for details)"
    echo "laplata_2025-01-14T13-48-20Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-01-14T13-48-20Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-02-17T14-16-27Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-02-17T14-16-27Z > "$LOG_DIR/laplata_2025-02-17T14-16-27Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-02-17T14-16-27Z crashed (Check log for details)"
    echo "laplata_2025-02-17T14-16-27Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-02-17T14-16-27Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-04-03T13-55-01Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-04-03T13-55-01Z > "$LOG_DIR/laplata_2025-04-03T13-55-01Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-04-03T13-55-01Z crashed (Check log for details)"
    echo "laplata_2025-04-03T13-55-01Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-04-03T13-55-01Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-05-11T14-04-26Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-05-11T14-04-26Z > "$LOG_DIR/laplata_2025-05-11T14-04-26Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-05-11T14-04-26Z crashed (Check log for details)"
    echo "laplata_2025-05-11T14-04-26Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-05-11T14-04-26Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-07-09T14-06-28Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-07-09T14-06-28Z > "$LOG_DIR/laplata_2025-07-09T14-06-28Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-07-09T14-06-28Z crashed (Check log for details)"
    echo "laplata_2025-07-09T14-06-28Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-07-09T14-06-28Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-08-23T14-23-16Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-08-23T14-23-16Z > "$LOG_DIR/laplata_2025-08-23T14-23-16Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-08-23T14-23-16Z crashed (Check log for details)"
    echo "laplata_2025-08-23T14-23-16Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-08-23T14-23-16Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-09-09T14-02-58Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-09-09T14-02-58Z > "$LOG_DIR/laplata_2025-09-09T14-02-58Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-09-09T14-02-58Z crashed (Check log for details)"
    echo "laplata_2025-09-09T14-02-58Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-09-09T14-02-58Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-09-11T14-11-36Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-09-11T14-11-36Z > "$LOG_DIR/laplata_2025-09-11T14-11-36Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-09-11T14-11-36Z crashed (Check log for details)"
    echo "laplata_2025-09-11T14-11-36Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-09-11T14-11-36Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-09-14T14-24-34Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-09-14T14-24-34Z > "$LOG_DIR/laplata_2025-09-14T14-24-34Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-09-14T14-24-34Z crashed (Check log for details)"
    echo "laplata_2025-09-14T14-24-34Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-09-14T14-24-34Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-11-23T14-12-23Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-11-23T14-12-23Z > "$LOG_DIR/laplata_2025-11-23T14-12-23Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-11-23T14-12-23Z crashed (Check log for details)"
    echo "laplata_2025-11-23T14-12-23Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-11-23T14-12-23Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-11-24T14-15-54Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-11-24T14-15-54Z > "$LOG_DIR/laplata_2025-11-24T14-15-54Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-11-24T14-15-54Z crashed (Check log for details)"
    echo "laplata_2025-11-24T14-15-54Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-11-24T14-15-54Z completed."
fi

echo "========================================="
echo "Processing: laplata_2025-12-18T14-01-25Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2025-12-18T14-01-25Z > "$LOG_DIR/laplata_2025-12-18T14-01-25Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2025-12-18T14-01-25Z crashed (Check log for details)"
    echo "laplata_2025-12-18T14-01-25Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2025-12-18T14-01-25Z completed."
fi

echo "========================================="
echo "Processing: laplata_2026-01-17T13-59-33Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2026-01-17T13-59-33Z > "$LOG_DIR/laplata_2026-01-17T13-59-33Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2026-01-17T13-59-33Z crashed (Check log for details)"
    echo "laplata_2026-01-17T13-59-33Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2026-01-17T13-59-33Z completed."
fi

echo "========================================="
echo "Processing: laplata_2026-01-18T14-02-32Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2026-01-18T14-02-32Z > "$LOG_DIR/laplata_2026-01-18T14-02-32Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2026-01-18T14-02-32Z crashed (Check log for details)"
    echo "laplata_2026-01-18T14-02-32Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2026-01-18T14-02-32Z completed."
fi

echo "========================================="
echo "Processing: laplata_2026-02-24T14-10-35Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2026-02-24T14-10-35Z > "$LOG_DIR/laplata_2026-02-24T14-10-35Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2026-02-24T14-10-35Z crashed (Check log for details)"
    echo "laplata_2026-02-24T14-10-35Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2026-02-24T14-10-35Z completed."
fi

echo "========================================="
echo "Processing: laplata_2026-03-25T13-49-23Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2026-03-25T13-49-23Z > "$LOG_DIR/laplata_2026-03-25T13-49-23Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2026-03-25T13-49-23Z crashed (Check log for details)"
    echo "laplata_2026-03-25T13-49-23Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2026-03-25T13-49-23Z completed."
fi

echo "========================================="
echo "Processing: laplata_2026-05-12T14-05-21Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2026-05-12T14-05-21Z > "$LOG_DIR/laplata_2026-05-12T14-05-21Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2026-05-12T14-05-21Z crashed (Check log for details)"
    echo "laplata_2026-05-12T14-05-21Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2026-05-12T14-05-21Z completed."
fi

echo "========================================="
echo "Processing: laplata_2026-05-13T14-07-34Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/laplata_2026-05-13T14-07-34Z > "$LOG_DIR/laplata_2026-05-13T14-07-34Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: laplata_2026-05-13T14-07-34Z crashed (Check log for details)"
    echo "laplata_2026-05-13T14-07-34Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: laplata_2026-05-13T14-07-34Z completed."
fi

echo "========================================="
echo "Processing: longisland_2025-01-22T15-57-39Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/longisland_2025-01-22T15-57-39Z > "$LOG_DIR/longisland_2025-01-22T15-57-39Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: longisland_2025-01-22T15-57-39Z crashed (Check log for details)"
    echo "longisland_2025-01-22T15-57-39Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: longisland_2025-01-22T15-57-39Z completed."
fi

echo "========================================="
echo "Processing: longisland_2025-01-24T16-10-41Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/longisland_2025-01-24T16-10-41Z > "$LOG_DIR/longisland_2025-01-24T16-10-41Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: longisland_2025-01-24T16-10-41Z crashed (Check log for details)"
    echo "longisland_2025-01-24T16-10-41Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: longisland_2025-01-24T16-10-41Z completed."
fi

echo "========================================="
echo "Processing: longisland_2025-03-27T16-07-43Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/longisland_2025-03-27T16-07-43Z > "$LOG_DIR/longisland_2025-03-27T16-07-43Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: longisland_2025-03-27T16-07-43Z crashed (Check log for details)"
    echo "longisland_2025-03-27T16-07-43Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: longisland_2025-03-27T16-07-43Z completed."
fi

echo "========================================="
echo "Processing: longisland_2025-08-08T16-06-57Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/longisland_2025-08-08T16-06-57Z > "$LOG_DIR/longisland_2025-08-08T16-06-57Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: longisland_2025-08-08T16-06-57Z crashed (Check log for details)"
    echo "longisland_2025-08-08T16-06-57Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: longisland_2025-08-08T16-06-57Z completed."
fi

echo "========================================="
echo "Processing: longisland_2025-12-05T16-09-54Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/longisland_2025-12-05T16-09-54Z > "$LOG_DIR/longisland_2025-12-05T16-09-54Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: longisland_2025-12-05T16-09-54Z crashed (Check log for details)"
    echo "longisland_2025-12-05T16-09-54Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: longisland_2025-12-05T16-09-54Z completed."
fi

echo "========================================="
echo "Processing: lucinda_2025-07-24T00-46-19Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/lucinda_2025-07-24T00-46-19Z > "$LOG_DIR/lucinda_2025-07-24T00-46-19Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: lucinda_2025-07-24T00-46-19Z crashed (Check log for details)"
    echo "lucinda_2025-07-24T00-46-19Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: lucinda_2025-07-24T00-46-19Z completed."
fi

echo "========================================="
echo "Processing: lucinda_2025-10-21T00-50-48Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/lucinda_2025-10-21T00-50-48Z > "$LOG_DIR/lucinda_2025-10-21T00-50-48Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: lucinda_2025-10-21T00-50-48Z crashed (Check log for details)"
    echo "lucinda_2025-10-21T00-50-48Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: lucinda_2025-10-21T00-50-48Z completed."
fi

echo "========================================="
echo "Processing: lucinda_2025-12-06T00-27-20Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/lucinda_2025-12-06T00-27-20Z > "$LOG_DIR/lucinda_2025-12-06T00-27-20Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: lucinda_2025-12-06T00-27-20Z crashed (Check log for details)"
    echo "lucinda_2025-12-06T00-27-20Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: lucinda_2025-12-06T00-27-20Z completed."
fi

echo "========================================="
echo "Processing: lucinda_2025-12-08T00-33-58Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/lucinda_2025-12-08T00-33-58Z > "$LOG_DIR/lucinda_2025-12-08T00-33-58Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: lucinda_2025-12-08T00-33-58Z crashed (Check log for details)"
    echo "lucinda_2025-12-08T00-33-58Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: lucinda_2025-12-08T00-33-58Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-02-05T15-52-26Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-02-05T15-52-26Z > "$LOG_DIR/mvco_2025-02-05T15-52-26Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-02-05T15-52-26Z crashed (Check log for details)"
    echo "mvco_2025-02-05T15-52-26Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-02-05T15-52-26Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-04-10T15-49-22Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-04-10T15-49-22Z > "$LOG_DIR/mvco_2025-04-10T15-49-22Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-04-10T15-49-22Z crashed (Check log for details)"
    echo "mvco_2025-04-10T15-49-22Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-04-10T15-49-22Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-04-30T15-59-40Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-04-30T15-59-40Z > "$LOG_DIR/mvco_2025-04-30T15-59-40Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-04-30T15-59-40Z crashed (Check log for details)"
    echo "mvco_2025-04-30T15-59-40Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-04-30T15-59-40Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-06-03T15-39-56Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-06-03T15-39-56Z > "$LOG_DIR/mvco_2025-06-03T15-39-56Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-06-03T15-39-56Z crashed (Check log for details)"
    echo "mvco_2025-06-03T15-39-56Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-06-03T15-39-56Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-06-24T15-46-22Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-06-24T15-46-22Z > "$LOG_DIR/mvco_2025-06-24T15-46-22Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-06-24T15-46-22Z crashed (Check log for details)"
    echo "mvco_2025-06-24T15-46-22Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-06-24T15-46-22Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-08-23T15-39-21Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-08-23T15-39-21Z > "$LOG_DIR/mvco_2025-08-23T15-39-21Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-08-23T15-39-21Z crashed (Check log for details)"
    echo "mvco_2025-08-23T15-39-21Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-08-23T15-39-21Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-09-11T15-27-40Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-09-11T15-27-40Z > "$LOG_DIR/mvco_2025-09-11T15-27-40Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-09-11T15-27-40Z crashed (Check log for details)"
    echo "mvco_2025-09-11T15-27-40Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-09-11T15-27-40Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-09-13T15-36-19Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-09-13T15-36-19Z > "$LOG_DIR/mvco_2025-09-13T15-36-19Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-09-13T15-36-19Z crashed (Check log for details)"
    echo "mvco_2025-09-13T15-36-19Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-09-13T15-36-19Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-11-02T15-47-30Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-11-02T15-47-30Z > "$LOG_DIR/mvco_2025-11-02T15-47-30Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-11-02T15-47-30Z crashed (Check log for details)"
    echo "mvco_2025-11-02T15-47-30Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-11-02T15-47-30Z completed."
fi

echo "========================================="
echo "Processing: mvco_2025-12-22T15-30-19Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2025-12-22T15-30-19Z > "$LOG_DIR/mvco_2025-12-22T15-30-19Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2025-12-22T15-30-19Z crashed (Check log for details)"
    echo "mvco_2025-12-22T15-30-19Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2025-12-22T15-30-19Z completed."
fi

echo "========================================="
echo "Processing: mvco_2026-01-20T15-24-27Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2026-01-20T15-24-27Z > "$LOG_DIR/mvco_2026-01-20T15-24-27Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2026-01-20T15-24-27Z crashed (Check log for details)"
    echo "mvco_2026-01-20T15-24-27Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2026-01-20T15-24-27Z completed."
fi

echo "========================================="
echo "Processing: mvco_2026-01-27T15-44-47Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2026-01-27T15-44-47Z > "$LOG_DIR/mvco_2026-01-27T15-44-47Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2026-01-27T15-44-47Z crashed (Check log for details)"
    echo "mvco_2026-01-27T15-44-47Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2026-01-27T15-44-47Z completed."
fi

echo "========================================="
echo "Processing: mvco_2026-02-24T15-26-32Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2026-02-24T15-26-32Z > "$LOG_DIR/mvco_2026-02-24T15-26-32Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2026-02-24T15-26-32Z crashed (Check log for details)"
    echo "mvco_2026-02-24T15-26-32Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2026-02-24T15-26-32Z completed."
fi

echo "========================================="
echo "Processing: mvco_2026-05-20T15-38-53Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/mvco_2026-05-20T15-38-53Z > "$LOG_DIR/mvco_2026-05-20T15-38-53Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: mvco_2026-05-20T15-38-53Z crashed (Check log for details)"
    echo "mvco_2026-05-20T15-38-53Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: mvco_2026-05-20T15-38-53Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-06-21T07-42-17Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-06-21T07-42-17Z > "$LOG_DIR/ngomeni_2025-06-21T07-42-17Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-06-21T07-42-17Z crashed (Check log for details)"
    echo "ngomeni_2025-06-21T07-42-17Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-06-21T07-42-17Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-06-22T07-47-04Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-06-22T07-47-04Z > "$LOG_DIR/ngomeni_2025-06-22T07-47-04Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-06-22T07-47-04Z crashed (Check log for details)"
    echo "ngomeni_2025-06-22T07-47-04Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-06-22T07-47-04Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-07-13T07-51-43Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-07-13T07-51-43Z > "$LOG_DIR/ngomeni_2025-07-13T07-51-43Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-07-13T07-51-43Z crashed (Check log for details)"
    echo "ngomeni_2025-07-13T07-51-43Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-07-13T07-51-43Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-07-14T07-56-25Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-07-14T07-56-25Z > "$LOG_DIR/ngomeni_2025-07-14T07-56-25Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-07-14T07-56-25Z crashed (Check log for details)"
    echo "ngomeni_2025-07-14T07-56-25Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-07-14T07-56-25Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-08-01T07-45-02Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-08-01T07-45-02Z > "$LOG_DIR/ngomeni_2025-08-01T07-45-02Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-08-01T07-45-02Z crashed (Check log for details)"
    echo "ngomeni_2025-08-01T07-45-02Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-08-01T07-45-02Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-11-02T07-58-06Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-11-02T07-58-06Z > "$LOG_DIR/ngomeni_2025-11-02T07-58-06Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-11-02T07-58-06Z crashed (Check log for details)"
    echo "ngomeni_2025-11-02T07-58-06Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-11-02T07-58-06Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-11-03T08-01-51Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-11-03T08-01-51Z > "$LOG_DIR/ngomeni_2025-11-03T08-01-51Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-11-03T08-01-51Z crashed (Check log for details)"
    echo "ngomeni_2025-11-03T08-01-51Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-11-03T08-01-51Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-11-04T08-05-36Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-11-04T08-05-36Z > "$LOG_DIR/ngomeni_2025-11-04T08-05-36Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-11-04T08-05-36Z crashed (Check log for details)"
    echo "ngomeni_2025-11-04T08-05-36Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-11-04T08-05-36Z completed."
fi

echo "========================================="
echo "Processing: ngomeni_2025-11-27T07-53-00Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/ngomeni_2025-11-27T07-53-00Z > "$LOG_DIR/ngomeni_2025-11-27T07-53-00Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ngomeni_2025-11-27T07-53-00Z crashed (Check log for details)"
    echo "ngomeni_2025-11-27T07-53-00Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: ngomeni_2025-11-27T07-53-00Z completed."
fi

echo "========================================="
echo "Processing: palgrunden_2025-03-01T21-14-45Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/palgrunden_2025-03-01T21-14-45Z > "$LOG_DIR/palgrunden_2025-03-01T21-14-45Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: palgrunden_2025-03-01T21-14-45Z crashed (Check log for details)"
    echo "palgrunden_2025-03-01T21-14-45Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: palgrunden_2025-03-01T21-14-45Z completed."
fi

echo "========================================="
echo "Processing: palgrunden_2025-03-15T10-04-30Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/palgrunden_2025-03-15T10-04-30Z > "$LOG_DIR/palgrunden_2025-03-15T10-04-30Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: palgrunden_2025-03-15T10-04-30Z crashed (Check log for details)"
    echo "palgrunden_2025-03-15T10-04-30Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: palgrunden_2025-03-15T10-04-30Z completed."
fi

echo "========================================="
echo "Processing: plocan_2025-08-17T12-02-50Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/plocan_2025-08-17T12-02-50Z > "$LOG_DIR/plocan_2025-08-17T12-02-50Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: plocan_2025-08-17T12-02-50Z crashed (Check log for details)"
    echo "plocan_2025-08-17T12-02-50Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: plocan_2025-08-17T12-02-50Z completed."
fi

echo "========================================="
echo "Processing: plocan_2025-11-16T11-54-16Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/plocan_2025-11-16T11-54-16Z > "$LOG_DIR/plocan_2025-11-16T11-54-16Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: plocan_2025-11-16T11-54-16Z crashed (Check log for details)"
    echo "plocan_2025-11-16T11-54-16Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: plocan_2025-11-16T11-54-16Z completed."
fi

echo "========================================="
echo "Processing: plocan_2026-01-09T11-42-06Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/plocan_2026-01-09T11-42-06Z > "$LOG_DIR/plocan_2026-01-09T11-42-06Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: plocan_2026-01-09T11-42-06Z crashed (Check log for details)"
    echo "plocan_2026-01-09T11-42-06Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: plocan_2026-01-09T11-42-06Z completed."
fi

echo "========================================="
echo "Processing: plocan_2026-03-25T11-56-04Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/plocan_2026-03-25T11-56-04Z > "$LOG_DIR/plocan_2026-03-25T11-56-04Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: plocan_2026-03-25T11-56-04Z crashed (Check log for details)"
    echo "plocan_2026-03-25T11-56-04Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: plocan_2026-03-25T11-56-04Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-02-02T09-06-35Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-02-02T09-06-35Z > "$LOG_DIR/section7platform_2025-02-02T09-06-35Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-02-02T09-06-35Z crashed (Check log for details)"
    echo "section7platform_2025-02-02T09-06-35Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-02-02T09-06-35Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-06-19T08-56-03Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-06-19T08-56-03Z > "$LOG_DIR/section7platform_2025-06-19T08-56-03Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-06-19T08-56-03Z crashed (Check log for details)"
    echo "section7platform_2025-06-19T08-56-03Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-06-19T08-56-03Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-07-13T09-15-10Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-07-13T09-15-10Z > "$LOG_DIR/section7platform_2025-07-13T09-15-10Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-07-13T09-15-10Z crashed (Check log for details)"
    echo "section7platform_2025-07-13T09-15-10Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-07-13T09-15-10Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-07-15T09-24-33Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-07-15T09-24-33Z > "$LOG_DIR/section7platform_2025-07-15T09-24-33Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-07-15T09-24-33Z crashed (Check log for details)"
    echo "section7platform_2025-07-15T09-24-33Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-07-15T09-24-33Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-07-29T08-54-36Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-07-29T08-54-36Z > "$LOG_DIR/section7platform_2025-07-29T08-54-36Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-07-29T08-54-36Z crashed (Check log for details)"
    echo "section7platform_2025-07-29T08-54-36Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-07-29T08-54-36Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-07-31T09-03-50Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-07-31T09-03-50Z > "$LOG_DIR/section7platform_2025-07-31T09-03-50Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-07-31T09-03-50Z crashed (Check log for details)"
    echo "section7platform_2025-07-31T09-03-50Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-07-31T09-03-50Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-08-17T08-46-06Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-08-17T08-46-06Z > "$LOG_DIR/section7platform_2025-08-17T08-46-06Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-08-17T08-46-06Z crashed (Check log for details)"
    echo "section7platform_2025-08-17T08-46-06Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-08-17T08-46-06Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-08-26T09-26-31Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-08-26T09-26-31Z > "$LOG_DIR/section7platform_2025-08-26T09-26-31Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-08-26T09-26-31Z crashed (Check log for details)"
    echo "section7platform_2025-08-26T09-26-31Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-08-26T09-26-31Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-09-13T09-10-10Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-09-13T09-10-10Z > "$LOG_DIR/section7platform_2025-09-13T09-10-10Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-09-13T09-10-10Z crashed (Check log for details)"
    echo "section7platform_2025-09-13T09-10-10Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-09-13T09-10-10Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-10-31T09-13-58Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-10-31T09-13-58Z > "$LOG_DIR/section7platform_2025-10-31T09-13-58Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-10-31T09-13-58Z crashed (Check log for details)"
    echo "section7platform_2025-10-31T09-13-58Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-10-31T09-13-58Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2025-11-22T08-58-57Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2025-11-22T08-58-57Z > "$LOG_DIR/section7platform_2025-11-22T08-58-57Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2025-11-22T08-58-57Z crashed (Check log for details)"
    echo "section7platform_2025-11-22T08-58-57Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2025-11-22T08-58-57Z completed."
fi

echo "========================================="
echo "Processing: section7platform_2026-02-19T08-47-28Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/section7platform_2026-02-19T08-47-28Z > "$LOG_DIR/section7platform_2026-02-19T08-47-28Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: section7platform_2026-02-19T08-47-28Z crashed (Check log for details)"
    echo "section7platform_2026-02-19T08-47-28Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: section7platform_2026-02-19T08-47-28Z completed."
fi

echo "========================================="
echo "Processing: skagerrak_2026-06-21T10-17-42Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/skagerrak_2026-06-21T10-17-42Z > "$LOG_DIR/skagerrak_2026-06-21T10-17-42Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: skagerrak_2026-06-21T10-17-42Z crashed (Check log for details)"
    echo "skagerrak_2026-06-21T10-17-42Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: skagerrak_2026-06-21T10-17-42Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-03-19T02-31-49Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-03-19T02-31-49Z > "$LOG_DIR/socheongcho_2025-03-19T02-31-49Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-03-19T02-31-49Z crashed (Check log for details)"
    echo "socheongcho_2025-03-19T02-31-49Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-03-19T02-31-49Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-03-20T02-37-37Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-03-20T02-37-37Z > "$LOG_DIR/socheongcho_2025-03-20T02-37-37Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-03-20T02-37-37Z crashed (Check log for details)"
    echo "socheongcho_2025-03-20T02-37-37Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-03-20T02-37-37Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-03-21T02-43-21Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-03-21T02-43-21Z > "$LOG_DIR/socheongcho_2025-03-21T02-43-21Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-03-21T02-43-21Z crashed (Check log for details)"
    echo "socheongcho_2025-03-21T02-43-21Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-03-21T02-43-21Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-04-03T02-21-02Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-04-03T02-21-02Z > "$LOG_DIR/socheongcho_2025-04-03T02-21-02Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-04-03T02-21-02Z crashed (Check log for details)"
    echo "socheongcho_2025-04-03T02-21-02Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-04-03T02-21-02Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-04-08T02-48-29Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-04-08T02-48-29Z > "$LOG_DIR/socheongcho_2025-04-08T02-48-29Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-04-08T02-48-29Z crashed (Check log for details)"
    echo "socheongcho_2025-04-08T02-48-29Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-04-08T02-48-29Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-05-29T02-25-27Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-05-29T02-25-27Z > "$LOG_DIR/socheongcho_2025-05-29T02-25-27Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-05-29T02-25-27Z crashed (Check log for details)"
    echo "socheongcho_2025-05-29T02-25-27Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-05-29T02-25-27Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-06-17T02-23-00Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-06-17T02-23-00Z > "$LOG_DIR/socheongcho_2025-06-17T02-23-00Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-06-17T02-23-00Z crashed (Check log for details)"
    echo "socheongcho_2025-06-17T02-23-00Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-06-17T02-23-00Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-06-22T02-47-05Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-06-22T02-47-05Z > "$LOG_DIR/socheongcho_2025-06-22T02-47-05Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-06-22T02-47-05Z crashed (Check log for details)"
    echo "socheongcho_2025-06-22T02-47-05Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-06-22T02-47-05Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-07-09T02-32-52Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-07-09T02-32-52Z > "$LOG_DIR/socheongcho_2025-07-09T02-32-52Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-07-09T02-32-52Z crashed (Check log for details)"
    echo "socheongcho_2025-07-09T02-32-52Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-07-09T02-32-52Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-07-11T02-42-19Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-07-11T02-42-19Z > "$LOG_DIR/socheongcho_2025-07-11T02-42-19Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-07-11T02-42-19Z crashed (Check log for details)"
    echo "socheongcho_2025-07-11T02-42-19Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-07-11T02-42-19Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-07-29T02-31-14Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-07-29T02-31-14Z > "$LOG_DIR/socheongcho_2025-07-29T02-31-14Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-07-29T02-31-14Z crashed (Check log for details)"
    echo "socheongcho_2025-07-29T02-31-14Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-07-29T02-31-14Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-08-23T02-49-48Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-08-23T02-49-48Z > "$LOG_DIR/socheongcho_2025-08-23T02-49-48Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-08-23T02-49-48Z crashed (Check log for details)"
    echo "socheongcho_2025-08-23T02-49-48Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-08-23T02-49-48Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-09-10T02-33-53Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-09-10T02-33-53Z > "$LOG_DIR/socheongcho_2025-09-10T02-33-53Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-09-10T02-33-53Z crashed (Check log for details)"
    echo "socheongcho_2025-09-10T02-33-53Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-09-10T02-33-53Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-11-24T02-42-54Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-11-24T02-42-54Z > "$LOG_DIR/socheongcho_2025-11-24T02-42-54Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-11-24T02-42-54Z crashed (Check log for details)"
    echo "socheongcho_2025-11-24T02-42-54Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-11-24T02-42-54Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2025-12-17T02-25-17Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2025-12-17T02-25-17Z > "$LOG_DIR/socheongcho_2025-12-17T02-25-17Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2025-12-17T02-25-17Z crashed (Check log for details)"
    echo "socheongcho_2025-12-17T02-25-17Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2025-12-17T02-25-17Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-01-16T02-23-46Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-01-16T02-23-46Z > "$LOG_DIR/socheongcho_2026-01-16T02-23-46Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-01-16T02-23-46Z crashed (Check log for details)"
    echo "socheongcho_2026-01-16T02-23-46Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-01-16T02-23-46Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-02-22T02-32-40Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-02-22T02-32-40Z > "$LOG_DIR/socheongcho_2026-02-22T02-32-40Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-02-22T02-32-40Z crashed (Check log for details)"
    echo "socheongcho_2026-02-22T02-32-40Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-02-22T02-32-40Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-02-23T02-35-20Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-02-23T02-35-20Z > "$LOG_DIR/socheongcho_2026-02-23T02-35-20Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-02-23T02-35-20Z crashed (Check log for details)"
    echo "socheongcho_2026-02-23T02-35-20Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-02-23T02-35-20Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-02-25T02-40-38Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-02-25T02-40-38Z > "$LOG_DIR/socheongcho_2026-02-25T02-40-38Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-02-25T02-40-38Z crashed (Check log for details)"
    echo "socheongcho_2026-02-25T02-40-38Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-02-25T02-40-38Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-03-04T02-58-58Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-03-04T02-58-58Z > "$LOG_DIR/socheongcho_2026-03-04T02-58-58Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-03-04T02-58-58Z crashed (Check log for details)"
    echo "socheongcho_2026-03-04T02-58-58Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-03-04T02-58-58Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-03-28T02-24-18Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-03-28T02-24-18Z > "$LOG_DIR/socheongcho_2026-03-28T02-24-18Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-03-28T02-24-18Z crashed (Check log for details)"
    echo "socheongcho_2026-03-28T02-24-18Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-03-28T02-24-18Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-04-02T02-36-26Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-04-02T02-36-26Z > "$LOG_DIR/socheongcho_2026-04-02T02-36-26Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-04-02T02-36-26Z crashed (Check log for details)"
    echo "socheongcho_2026-04-02T02-36-26Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-04-02T02-36-26Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-04-07T02-48-13Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-04-07T02-48-13Z > "$LOG_DIR/socheongcho_2026-04-07T02-48-13Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-04-07T02-48-13Z crashed (Check log for details)"
    echo "socheongcho_2026-04-07T02-48-13Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-04-07T02-48-13Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-04-08T02-50-34Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-04-08T02-50-34Z > "$LOG_DIR/socheongcho_2026-04-08T02-50-34Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-04-08T02-50-34Z crashed (Check log for details)"
    echo "socheongcho_2026-04-08T02-50-34Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-04-08T02-50-34Z completed."
fi

echo "========================================="
echo "Processing: socheongcho_2026-05-22T02-54-54Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/socheongcho_2026-05-22T02-54-54Z > "$LOG_DIR/socheongcho_2026-05-22T02-54-54Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: socheongcho_2026-05-22T02-54-54Z crashed (Check log for details)"
    echo "socheongcho_2026-05-22T02-54-54Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: socheongcho_2026-05-22T02-54-54Z completed."
fi

echo "========================================="
echo "Processing: wilmington_2025-07-14T15-48-27Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/wilmington_2025-07-14T15-48-27Z > "$LOG_DIR/wilmington_2025-07-14T15-48-27Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: wilmington_2025-07-14T15-48-27Z crashed (Check log for details)"
    echo "wilmington_2025-07-14T15-48-27Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: wilmington_2025-07-14T15-48-27Z completed."
fi

echo "========================================="
echo "Processing: wilmington_2025-12-27T15-48-39Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/wilmington_2025-12-27T15-48-39Z > "$LOG_DIR/wilmington_2025-12-27T15-48-39Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: wilmington_2025-12-27T15-48-39Z crashed (Check log for details)"
    echo "wilmington_2025-12-27T15-48-39Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: wilmington_2025-12-27T15-48-39Z completed."
fi

echo "========================================="
echo "Processing: wilmington_2026-03-17T16-22-48Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/wilmington_2026-03-17T16-22-48Z > "$LOG_DIR/wilmington_2026-03-17T16-22-48Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: wilmington_2026-03-17T16-22-48Z crashed (Check log for details)"
    echo "wilmington_2026-03-17T16-22-48Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: wilmington_2026-03-17T16-22-48Z completed."
fi

echo "========================================="
echo "Processing: zeebrugge_2025-03-08T11-00-54Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/zeebrugge_2025-03-08T11-00-54Z > "$LOG_DIR/zeebrugge_2025-03-08T11-00-54Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: zeebrugge_2025-03-08T11-00-54Z crashed (Check log for details)"
    echo "zeebrugge_2025-03-08T11-00-54Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: zeebrugge_2025-03-08T11-00-54Z completed."
fi

echo "========================================="
echo "Processing: zeebrugge_2025-08-09T11-19-43Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/zeebrugge_2025-08-09T11-19-43Z > "$LOG_DIR/zeebrugge_2025-08-09T11-19-43Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: zeebrugge_2025-08-09T11-19-43Z crashed (Check log for details)"
    echo "zeebrugge_2025-08-09T11-19-43Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: zeebrugge_2025-08-09T11-19-43Z completed."
fi

echo "========================================="
echo "Processing: zeebrugge_2025-09-01T11-27-47Z"
echo "========================================="
python /home/camerop/AC/hypso-ac-processing/2b_process_capture.py /home/camerop/HYPSO_DATA_AOC/zeebrugge_2025-09-01T11-27-47Z > "$LOG_DIR/zeebrugge_2025-09-01T11-27-47Z.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ FAILED: zeebrugge_2025-09-01T11-27-47Z crashed (Check log for details)"
    echo "zeebrugge_2025-09-01T11-27-47Z" >> "$LOG_DIR/failed_captures.txt"
else
    echo "✅ SUCCESS: zeebrugge_2025-09-01T11-27-47Z completed."
fi

