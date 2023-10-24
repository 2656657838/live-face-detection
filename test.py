import requests

import base64
import time

# 那个images就传base64编码的列表
# def getByte(path):
#     with open(path, 'rb') as f:
#         img_byte = base64.b64encode(f.read())
#     img_str = img_byte.decode('utf-8')
#     return img_str
# 然后是[img_str1,img_str2]

# 设置接口的基本URL
base_url = "http://localhost:12727"
def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('utf-8')
    return img_str

def test_receive_signal():
    # 准备测试数据，这里假设发送一个 JSON 数据作为请求体

    images=[]
    for i in range(1,30):
        img_str = getByte(f'data/nod/image_{i}.jpg')
        images.append(img_str)
    # images.append(getByte('test.png'))
    


    data = {
        "images": images,
        "action": 'blink'
    }

    # 发送 POST 请求到接口
    time.sleep(0.01)
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Content-Type': 'application/json'
    }

    response = requests.post(f"{base_url}/face-alive-detect", headers=headers, json=data)

    
    # 检查响应状态码
    if response.status_code == 200:
        print("Test passed: Status code is 200")
    else:
        print(f"Test failed: Status code is {response.status_code}")

    # 检查响应内容，这里假设接口返回 "Signal received"
    if response.text == "Signal received":
        print("Test passed: Response content is 'Signal received'")
    else:
        print(response.text)

if __name__ == '__main__':
    test_receive_signal()
