# live-face-detection

## action
    1.左摇头
    2.右摇头
    3.点头
    4.嘴部动作
## 参数
    |参数名|是否必选|参数类型|说明|
    |nod_threshold|否|double|该参数为点头动作幅度的判断门限,取值范围:[1,90],默认为10,单位为度。该值设置越大,则越难判断为点头|
    video_url

## 返回
    1.alive bool
    2.actions array
    4.confidence doubel

## 接口响应样例
状态码 200
    {
    "video-result": {
        "alive": true,
        "actions": [
        {
            "confidence": 0.823,
            "action": 1
        },
        {
            "confidence": 0.823,
            "action": 3
        },
        {
            "confidence": 0.823,
            "action": 2
        }
        ],
        "picture": "/9j/4AAQSkZJRgABAQEAYABgAAD/2w..."
    },
    "warning-list": []
    }
