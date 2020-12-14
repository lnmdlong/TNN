/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <jni.h>

#include <sstream>
#include <string>

#include "benchmark_model_jni.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

JNIEXPORT void JNICALL TNN_BENCHMARK_MODEL(nativeRun)(JNIEnv* env, jobject thiz, jstring args_obj) {
  const char* args_chars = env->GetStringUTFChars(args_obj, nullptr);

  // Split the args string into individual arg tokens.
  std::istringstream iss(args_chars);
  std::vector<std::string> args_split{std::istream_iterator<std::string>(iss),
                                      {}};

  // Construct a fake argv command-line object for the benchmark.
  std::vector<char*> argv;
  std::string arg0 = "(BenchmarkModelAndroid)";
  argv.push_back(const_cast<char*>(arg0.data()));
  for (auto& arg : args_split) {
    argv.push_back(const_cast<char*>(arg.data()));
  }

  char ** argv_data;

  TNN_NS::benchmark::Run(static_cast<int>(argv.size()), argv_data);

  env->ReleaseStringUTFChars(args_obj, args_chars);
}
