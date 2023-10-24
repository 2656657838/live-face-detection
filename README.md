# live-face-detection

## action
    1.静默检测
    2.眨眼
    3.张嘴
    4.摇头
    5.点头
## 参数

```python
    {
        "images": list # [base64.b64encode(f.read()).decode('utf-8'),...]
        "action": str # 'nothing', 'blink', 'gape', 'shake', 'nod'
    }



```

## 返回
    ```python
    {
        "action":"nothing",
        "alive":false,
        "conf":0,
        "is_error":false,
        "msg":null
    }
    
    ```

