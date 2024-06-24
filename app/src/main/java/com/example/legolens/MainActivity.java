package com.example.legolens;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.example.legolens.ml.Model;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 64;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 1}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 1);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            // Iterate over each pixel and extract the grayscale value. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    // Convert the RGB value to grayscale
                    int gray = (int)(((val >> 16) & 0xFF) * 0.299 + ((val >> 8) & 0xFF) * 0.587 + (val & 0xFF) * 0.114);
                    // Normalize the value and add to the byte buffer
                    byteBuffer.putFloat(gray);
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"10247", "11090", "11211", "11212", "11214", "11458", "11476", "11477", "14704", "14719", "14769", "15068", "15070", "15100", "15379", "15392", "15535", "15573", "15712", "18651", "18654", "18674", "18677", "20482", "22388", "22885", "2357", "2412b", "2420", "24201", "24246", "2429", "2430", "2431", "2432", "2436", "2445", "2450", "2454", "2456", "24866", "25269", "2540", "26047", "2654", "26601", "26603", "26604", "2780", "27925", "28192", "2877", "3001", "3002", "3003", "3004", "3005", "3008", "3009", "3010", "30136", "3020", "3021", "3022", "3023", "3024", "3031", "3032", "3034", "3035", "3037", "30374", "3039", "3040", "30413", "30414", "3062b", "3065", "3068b", "3069b", "3070b", "32000", "32013", "32028", "32054", "32062", "32064", "32073", "32123", "32140", "32184", "32278", "32316", "3245c", "32523", "32524", "32525", "32526", "32607", "32952", "33291", "33909", "34103", "3460", "35480", "3622", "3623", "3660", "3665", "3666", "3673", "3700", "3701", "3705", "3710", "3713", "3749", "3795", "3832", "3937", "3941", "3958", "4032", "40490", "4070", "4073", "4081b", "4085", "4162", "41677", "41740", "41769", "41770", "42003", "4274", "4286", "43093", "43722", "43723", "44728", "4477", "4519", "4589", "4599b", "4740", "47457", "48336", "4865", "48729", "49668", "50950", "51739", "53451", "54200", "59443", "60470", "60474", "60478", "60479", "60481", "60483", "60592", "60601", "6091", "61252", "6134", "61409", "61678", "62462", "63864", "63868", "63965", "64644", "6536", "6541", "6558", "6632", "6636", "85080", "85861", "85984", "87079", "87083", "87087", "87552", "87580", "87620", "87994", "88072", "88323", "92280", "92946", "93273", "98138", "98283", "99206", "99207", "99563", "99780", "99781"};
            String s = "Lego ";
            s += classes[maxPos];
            s += "\n"+maxConfidence+"%";

            result.setText(s);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                imageView.setImageBitmap(image);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}