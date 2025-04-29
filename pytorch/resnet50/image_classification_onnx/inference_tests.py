import subprocess
import csv
import re


CSV_FILE = 'inference_times.csv'

def extract_inference_time(output):
    match = re.search(r'Inference Time = ([\d.]+) ms', output)
    if match:
        return float(match.group(1))
    return None

def test_inference_script(precision):
    result = subprocess.run([
    'python3', 'inference_onnx.py', '--ep', 'rocm', '--precision', precision
    ], capture_output=True, text=True)
    print(f"Precision: {precision}")
    print(result.stdout)
    if result.returncode != 0:
        print("Error (stderr):", result.stderr)
    assert result.returncode == 0
    inference_time = extract_inference_time(result.stdout)
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([precision, inference_time])
    return inference_time

def convert_onnx(precision):
    print("Converting precision: ", precision)
    result = subprocess.run([
        'python3', 'conversion.py', '--precision', precision
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("conversion.py failed!")
        print("STDOUT:\n", result.stdout)
        print("STDERR (traceback):\n", result.stderr)
    else:
        print("conversion.py output:\n", result.stdout)
    return

if __name__ == '__main__':
    
    convert_onnx("fp32")
    convert_onnx("fp16")
    convert_onnx("mixed")



    # Write CSV header
    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['precision', 'inference_time_ms'])
    results = {'fp32': [], 'fp16': [], 'mixed': []}
    for i in range(3):
        t = test_inference_script("fp32")
        results['fp32'].append(t)
    for i in range(3):
        t = test_inference_script("fp16")
        results['fp16'].append(t)
    for i in range(3):
        t = test_inference_script("mixed")
        results['mixed'].append(t)
    # Write averages to CSV
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for precision, times in results.items():
            valid_times = [t for t in times if t is not None]
            avg = sum(valid_times)/len(valid_times) if valid_times else None
            writer.writerow([f'{precision}_average', avg])