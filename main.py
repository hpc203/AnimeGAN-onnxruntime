#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
import onnxruntime as ort

class AnimeGAN():
    def __init__(self):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession('face_paint_512_v2_0.onnx', so)
        self.input_size = 512
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name
    def detect(self, image):
        img = cv2.resize(image, (self.input_size, self.input_size))
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = x.transpose(2, 0, 1).astype('float32')
        x = x * 2 - 1
        x = x.reshape(-1, 3, self.input_size, self.input_size)

        outs = self.net.run([self.output_name], {self.input_name: x})[0].squeeze(axis=0)
        outs = (outs * 0.5 + 0.5).clip(0, 1)
        outs = outs * 255
        outs = outs.transpose(1, 2, 0).astype('uint8')
        outs = cv2.cvtColor(outs, cv2.COLOR_RGB2BGR)
        return cv2.resize(outs, (image.shape[1], image.shape[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='sample.jpg')
    args = parser.parse_args()

    model = AnimeGAN()
    srcimg = cv2.imread(args.imgpath)
    result = model.detect(srcimg)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', srcimg)
    winName = 'Deep learning AnimeGAN in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()