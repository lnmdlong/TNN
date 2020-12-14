package com.tencent.tnn.demo;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Debug;
import android.view.View;
import android.widget.TextView;
import android.util.Log;
import com.tencent.tnn.demo.BenchmarkModel;


public class MainActivity extends Activity {

    private TextView lightLiveCheckBtn;

    private boolean isShowedActivity = false;

    private static final String TAG = "TNN_BenchmarkModelActivity";

    private static final String ARGS_INTENT_KEY_0 = "args";
    private static final String ARGS_INTENT_KEY_1 = "--args";
    private BenchmarkModel benchmark = new BenchmarkModel();

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        System.loadLibrary("tnn_wrapper");
        Intent intent = getIntent();
        Bundle bundle = intent.getExtras();
        String args = "";

        Log.i(TAG, "Running TNN Benchmark with args: " + args);
        benchmark.nativeRun(args);
    }

    @Override
    protected void onResume() {
        super.onResume();
        isShowedActivity = false;
    }

}
