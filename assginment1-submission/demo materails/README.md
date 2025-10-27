# Running Instructions

## Part 1: TensorFlow

1. Install required packages:
   ```bash
   pip install tensorflow numpy
   ```
2. Run the script:
   ```bash
   python part1_tensorflow.py
   ```

## Part 2: TensorFlow Lite

1. Install required packages:
   ```bash
   pip install tensorflow numpy
   ```
2. Run the script:
   ```bash
   python part2_tensorflowLite.py
   ```
3. Modify the `.keras` model file path to the current location of your model.

## Part 3: TensorFlow Lite Micro

1. Modify the `TFLITE_MICRO_LIB` path in the Makefile to point to the correct location.
2. Install OpenCV using Homebrew:
   ```bash
   brew install opencv
   ```
3. Build the project using Makefile:
   ```bash
   make
   ```
4. This will generate the executable file:
   ```bash
   ./model_inference
   ```
5. Make sure to adjust the `test_image` path to the appropriate location.
