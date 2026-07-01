#!/bin/bash

# Enable Python fault handler to catch segfault lines
export PYTHONFAULTHANDLER=1

# Create a log directory
LOG_DIR="/home/camerop/AC/logs"
mkdir -p "$LOG_DIR"



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

