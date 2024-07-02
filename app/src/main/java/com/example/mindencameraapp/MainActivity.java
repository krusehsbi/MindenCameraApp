/**
 * ---------------------------------------------------------------------------
 * Vorlesung: Deep Learning for Computer Vision (SoSe 2024)
 * Thema:     Test App for CameraX & TensorFlow Lite
 *
 * @author Jan Rexilius
 * @date   02/2024
 * ---------------------------------------------------------------------------
 */

package com.example.mindencameraapp;

import androidx.annotation.NonNull;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.media.ImageReader;
import android.util.Log;
import android.util.Size;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.VideoCapture;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;


// ----------------------------------------------------------------------
// main class
public class MainActivity extends AppCompatActivity implements ImageAnalysis.Analyzer {

    private static final String TAG = "LOGGING:";
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    private ImageAnalysis imageAnalyzer;
    // image buffer
    private Bitmap bitmapBuffer;

    ImageButton buttonTakePicture;
    ImageButton buttonGallery;
    TextView classificationResults;
    private ImageCapture imageCapture;
    private VideoCapture videoCapture;
    private ExecutorService cameraExecutor;

    private int REQUEST_CODE_PERMISSIONS = 10;
    private final String[] REQUIRED_PERMISSIONS = new String[]{
            "android.permission.CAMERA"
    };

    private final Object task = new Object();

    private Interpreter tflite;
    // add your filename here (label names)
    final String CLASSIFIER_LABEL_File = "alphabet_labels.txt";
    // add your filename here (model file)
    final String TF_LITE_File = "student_model_mobilenet_10.tflite";
    List<String> clasifierLabels = null;




    // ----------------------------------------------------------------------
    // set gui elements and start workflow
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classificationResults = findViewById(R.id.classificationResults);
        buttonTakePicture = findViewById(R.id.buttonCapture);
        buttonTakePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureImage();
            }
        });
        buttonGallery = findViewById(R.id.buttonGallery);
        buttonGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveImage();
            }
        });

        previewView = findViewById(R.id.previewView);

        // Check and request permissions if needed
        if (!allPermissionsGranted()) {
            requestPermissionsIfNeeded();
        } else {
            // Permissions are already granted, proceed with your app logic
            Toast.makeText(this, "Permissions already granted", Toast.LENGTH_SHORT).show();
            // Proceed with your app logic here
            cameraProviderFuture = ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(() -> {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    startCameraX(cameraProvider);
                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }, getExecutor());
        }
    }

    // ----------------------------------------------------------------------
    // check app permissions
    private void checkPermissions() {
        if (allPermissionsGranted()) {
            cameraProviderFuture = ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(() -> {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    startCameraX(cameraProvider);
                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }, getExecutor());
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void requestPermissionsIfNeeded() {
        List<String> permissionsToRequest = new ArrayList<>();
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(permission);
            }
        }

        if (!permissionsToRequest.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissionsToRequest.toArray(new String[0]), REQUEST_CODE_PERMISSIONS);
        } else {
            // All permissions are already granted
            Toast.makeText(this, "Permissions already granted", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "All permissions already granted");
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            Log.d(TAG, "onRequestPermissionsResult: Received permission result");

            // Check if all permissions were granted
            boolean allPermissionsGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allPermissionsGranted = false;
                    break;
                }
            }

            if (allPermissionsGranted) {
                Log.d(TAG, "onRequestPermissionsResult: All permissions granted");
                Toast.makeText(this, "Permissions granted", Toast.LENGTH_SHORT).show();
                // Proceed with your app logic here
            } else {
                Log.d(TAG, "onRequestPermissionsResult: Permissions not granted");
                Toast.makeText(this, "Permissions not granted", Toast.LENGTH_SHORT).show();
                // Handle denied permissions, possibly show rationale or disable functionality
            }

            // Log which permissions were not granted
            for (int i = 0; i < permissions.length; i++) {
                if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                    Log.d(TAG, "Permission " + permissions[i] + " was not granted");
                }
            }
        }
    }

    private Executor getExecutor() {
        return ContextCompat.getMainExecutor(this);
    }


    // ----------------------------------------------------------------------
    // start camera
    @SuppressLint("RestrictedApi")
    private void startCameraX(ProcessCameraProvider cameraProvider) {

        // load label file
        try {
            clasifierLabels = FileUtil.loadLabels(this, CLASSIFIER_LABEL_File);
        } catch (IOException e) {
            Log.e("startCameraX", "Error reading label file", e);
        }
        // load tf lite model
        try{
            MappedByteBuffer tfliteModel;
            tfliteModel = FileUtil.loadMappedFile(this,TF_LITE_File);
            tflite = new Interpreter(tfliteModel);

        } catch (IOException e){
            Log.e("startCameraX", "Error reading model", e);
        }


        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setFlashMode(ImageCapture.FLASH_MODE_AUTO)
                .build();

        imageAnalyzer = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(previewView.getDisplay().getRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();
        imageAnalyzer.setAnalyzer(getExecutor(), this);

        // unbind before binding
        cameraProvider.unbindAll();
        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalyzer, preview, imageCapture);
        } catch (Exception exc) {
            Log.e(TAG, "Use case binding failed", exc);
        }
    }


    // ----------------------------------------------------------------------
    // capture single image
    private void captureImage() {
        imageCapture.takePicture(getExecutor(), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                super.onCaptureSuccess(image);
                Log.d("TAG", "Capture Image");
                classifySingleImage(image);
            }
        });
    }


    // ----------------------------------------------------------------------
    // save image to file
    private void saveImage() {
        long timeStamp = System.currentTimeMillis();
        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, timeStamp);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");

        imageCapture.takePicture(
                new ImageCapture.OutputFileOptions.Builder(
                        getContentResolver(),
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                        contentValues
                ).build(),
                getExecutor(),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        Toast.makeText(MainActivity.this,"Saving...",Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Toast.makeText(MainActivity.this,"Error: "+exception.getMessage(),Toast.LENGTH_SHORT).show();


                    }
                });
    }

    // ----------------------------------------------------------------------
    // classify single image
    private void classifySingleImage(@NonNull ImageProxy imageProxy) {
        Log.d("classifySingleImage", "CLASSIFY_IMAGE " + imageProxy.getImageInfo().getTimestamp());

        Log.d("analyze", "format " + imageProxy.getFormat());

        ByteBuffer buffer = imageProxy.getPlanes()[0].getBuffer();
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        Bitmap bitmapImage = BitmapFactory.decodeByteArray(arr, 0, buffer.capacity());
        imageProxy.close();

        int rotation = imageProxy.getImageInfo().getRotationDegrees();

        imageProxy.close();
        int width  = bitmapImage.getWidth();
        int height = bitmapImage.getHeight();
        Log.d("classifySingleImage", "(width,height): " + width + " " + height);

        // image size set to 224x224 (use bilinear interpolation)
        int size = height > width ? width : height;
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new Rot90Op(1))
                .add(new ResizeWithCropOrPadOp(size, size))
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build();


        TensorImage tensorImage = new TensorImage(DataType.UINT8);
        tensorImage.load(bitmapImage);
        tensorImage = imageProcessor.process(tensorImage);
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 27}, DataType.UINT8);

        if(null != tflite) {
            tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
        }
        TensorProcessor probabilityProcessor =
                new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

        String resultString = " ";
        if (null != clasifierLabels) {
            // Map of labels and their corresponding probability
            TensorLabel labels = new TensorLabel(clasifierLabels,
                    probabilityProcessor.process(probabilityBuffer));

            // Create a map to access the result based on label
            Map<String, Float> floatMap = labels.getMapWithFloatValue();
            resultString = getResultString(floatMap);
            Log.d("classifySingleImage", "RESULT: " + resultString);
            Toast.makeText(MainActivity.this, resultString, Toast.LENGTH_SHORT).show();
        }
    }


    private Bitmap yuvImageToBitmap(Image image) {
        // input image is in yuv-format
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 90, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }


    // ----------------------------------------------------------------------
    // process current frame
    @Override
    public void analyze(@NonNull ImageProxy imageProxy) {
        if ( imageProxy.getFormat()== PixelFormat.RGBA_8888){
            Bitmap bitmapImage = Bitmap.createBitmap(imageProxy.getWidth(),imageProxy.getHeight(),Bitmap.Config.ARGB_8888);
            bitmapImage.copyPixelsFromBuffer(imageProxy.getPlanes()[0].getBuffer());

            int rotation = imageProxy.getImageInfo().getRotationDegrees();

            imageProxy.close();
            int width  = bitmapImage.getWidth();
            int height = bitmapImage.getHeight();

            int size = height > width ? width : height;
            // image size set to 224x224 (use bilinear interpolation)
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new Rot90Op(1))
                    .add(new ResizeWithCropOrPadOp(size, size))
                    .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .build();

            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmapImage);
            tensorImage = imageProcessor.process(tensorImage);
            TensorBuffer probabilityBuffer =
                    TensorBuffer.createFixedSize(new int[]{1, 27}, DataType.FLOAT32);

            if(null != tflite) {
                tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
            }
            TensorProcessor probabilityProcessor =
                    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

            String resultString = " ";
            if (null != clasifierLabels) {

                Log.d(TAG, "Labels size: " + clasifierLabels.size());
                // Log the shape dimensions
                int[] shape = probabilityBuffer.getShape();
                StringBuilder shapeStringBuilder = new StringBuilder("Probabilities shape: [");
                for (int dim : shape) {
                    shapeStringBuilder.append(dim).append(", ");
                }
                shapeStringBuilder.append("]");
                Log.d(TAG, shapeStringBuilder.toString());


                // Map of labels and their corresponding probability
                TensorLabel labels = new TensorLabel(clasifierLabels,
                        probabilityProcessor.process(probabilityBuffer));
                // Create a map to access the result based on label
                Map<String, Float> floatMap = labels.getMapWithFloatValue();
                resultString = getBestResult(floatMap);
                //Log.d("classifyImage", "RESULT: " + resultString);
                classificationResults.setText(resultString);
                //Toast.makeText(MainActivity.this, resultString, Toast.LENGTH_SHORT).show();
            }
        }
        // close image to get next one
        imageProxy.close();
    }


    // ----------------------------------------------------------------------
    // get 3 best keys & values from TF results
    public static String getResultString(Map<String, Float> mapResults){
        // max value
        Map.Entry<String, Float> entryMax1 = null;
        // 2nd max value
        Map.Entry<String, Float> entryMax2 = null;
        // 3rd max value
        Map.Entry<String, Float> entryMax3 = null;
        for(Map.Entry<String, Float> entry: mapResults.entrySet()){
            if (entryMax1 == null || entry.getValue().compareTo(entryMax1.getValue()) > 0){
                entryMax1 = entry;
            } else if (entryMax2 == null || entry.getValue().compareTo(entryMax2.getValue()) > 0){
                entryMax2 = entry;
            } else if (entryMax3 == null || entry.getValue().compareTo(entryMax3.getValue()) > 0){
                entryMax3 = entry;
            }
        }
        // result string includes the first three best values
        String result = entryMax1.getKey().trim() + " " + entryMax1.getValue().toString() + "\n" +
                        entryMax2.getKey().trim() + " " + entryMax2.getValue().toString() + "\n" +
                        entryMax3.getKey().trim() + " " + entryMax3.getValue().toString() + "\n";
        return result;
    }


    // ----------------------------------------------------------------------
    // get best key & value from TF results
    public static String getBestResult(@NonNull Map<String, Float> mapResults){
        // max value
        Map.Entry<String, Float> entryMax = null;
        for(Map.Entry<String, Float> entry: mapResults.entrySet()){
            if (entryMax == null || entry.getValue().compareTo(entryMax.getValue()) > 0) {
                entryMax = entry;
            }
        }
        int val = (int)(entryMax.getValue()*100.0f);
        entryMax.setValue((float)val);
        // result string includes the first three best values
        String result = "  " + entryMax.getKey().trim() + "   (" + Integer.toString(val) + "%)";
        return result;
    }

} // class