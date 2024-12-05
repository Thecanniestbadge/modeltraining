package com.example.modeltraining;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    // Constant defining the size of images the model expects
    private static final int IMG_SIZE = 224;

    // UI elements
    private ImageView imageView; // Displays the selected image
    private TextView textViewResult; // Displays the predicted class
    private TextView textViewLabel; // Displays the label of the predicted class

    // TensorFlow Lite interpreter for running the model
    private Interpreter tflite;

    // Bitmap for the selected image
    private Bitmap selectedBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        imageView = findViewById(R.id.imageView);
        textViewResult = findViewById(R.id.textView_result);
        textViewLabel = findViewById(R.id.textView_label); // Initialize TextView for labels
        RadioGroup radioGroup = findViewById(R.id.radioGroup); // Group for radio buttons
        Button buttonAnalyze = findViewById(R.id.button_analyze); // Button to trigger analysis

        // Load the TensorFlow Lite model
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace(); // Handle model loading errors
        }

        // Set listener for radio button group to display different images
        radioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.radio_image1) {
                selectedBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.image1); // Load first image
            } else if (checkedId == R.id.radio_image2) {
                selectedBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.image2); // Load second image
            } else if (checkedId == R.id.radio_image3) {
                selectedBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.image3); // Load third image
            }
            imageView.setImageBitmap(selectedBitmap); // Show the selected image in the ImageView
        });

        // Set listener for the "Analyze" button
        buttonAnalyze.setOnClickListener(v -> {
            if (selectedBitmap != null) {
                runModelInference(selectedBitmap); // Run model inference on the selected image
            } else {
                textViewResult.setText("Please select an image first."); // Display a message if no image is selected
            }
        });
    }

    // Load the TensorFlow Lite model file from assets
    private MappedByteBuffer loadModelFile() throws IOException {
        try (FileInputStream inputStream = new FileInputStream(getAssets().openFd("optimized_image_recognition_model.tflite").getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = getAssets().openFd("optimized_image_recognition_model.tflite").getStartOffset();
            long declaredLength = getAssets().openFd("optimized_image_recognition_model.tflite").getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    // Run inference using the TensorFlow Lite model on the selected image
    private void runModelInference(Bitmap bitmap) {
        // Resize the bitmap to the model's expected input size
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true);

        // Prepare input buffer to store the image data
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * IMG_SIZE * IMG_SIZE * 3); // 4 bytes per float, 3 channels
        inputBuffer.order(ByteOrder.nativeOrder()); // Use the device's native byte order

        // Normalize image pixel values and put them in the buffer
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                int px = resizedBitmap.getPixel(x, y); // Get pixel value
                inputBuffer.putFloat(((px >> 16) & 0xFF) / 255.0f); // Red channel
                inputBuffer.putFloat(((px >> 8) & 0xFF) / 255.0f); // Green channel
                inputBuffer.putFloat((px & 0xFF) / 255.0f); // Blue channel
            }
        }

        // Prepare output buffer for the model's predictions
        float[][] output = new float[1][getNumberOfClasses()];

        // Run inference
        tflite.run(inputBuffer, output);

        // Get the class with the highest confidence score
        int predictedClass = argMax(output[0]);
        textViewResult.setText("Predicted Class: " + predictedClass); // Display predicted class

        // Get the label for the predicted class and display it
        String label = getLabelForPrediction(predictedClass);
        textViewLabel.setText("Label: " + label); // Show label
    }

    // Helper method to find the index of the highest value in an array
    private int argMax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex; // Return the index of the maximum value
    }

    // Return the number of classes the model can predict
    private int getNumberOfClasses() {
        return 11; // Adjust based on the model's class count
    }

    // Map prediction indices to readable labels
    private String getLabelForPrediction(int prediction) {
        // Create a map of class to labels
        Map<Integer, String> labels = new HashMap<>();
        labels.put(1, "Alligator");
        labels.put(2, "Bass");
        labels.put(3, "Crocodile");
        labels.put(4, "Axolotl");
        labels.put(5, "Frog");
        labels.put(6, "Goldfish");
        labels.put(7, "Hammerhead Shark");
        labels.put(8, "Sea Turtle");
        labels.put(9, "Great White Shark");
        labels.put(10, "Stingray");
        labels.put(11, "Turtle");

        // Return the corresponding label or "Unknown" if the index is not mapped
        return labels.getOrDefault(prediction, "Unknown");
    }
}
