depend:
	brew install opencv
	brew install onnxruntime
build:
	clang++ -std=c++14 `pkg-config --cflags --libs  opencv4` `pkg-config --cflags --libs libonnxruntime` -o demo main.cpp
run:
	./demo