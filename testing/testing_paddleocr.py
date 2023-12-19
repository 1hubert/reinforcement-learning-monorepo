from paddleocr import PaddleOCR
import time

ocr = PaddleOCR(
    lang='en',
    rec_char_dict_path='./allowed_chars.txt'
)
img_path = './test.png' #define your image path here

for _ in range(3):
    start = time.time()
    text = ocr.ocr(img_path, det=False, cls=False)
    end = time.time()
    print(f"Time taken: {end-start} sec")
    print(text)
    print(text[0][0])
