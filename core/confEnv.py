def modify_line_in_file(file_path, line_number, code, new_code):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if line_number <= len(lines):
            if str(code) in lines[line_number - 1]:
                print('Environment no configured!')
                lines[line_number - 1] = new_code + '\n'
            else:
                print('Environment configured successfully')
                return 0

            with open(file_path, 'w') as file:
                file.writelines(lines)
            print(f'{line_number}:changed')
        else:
            print(f'no {line_number} line')
    except Exception as e:
        print(f'error：{str(e)}')

def confEnv(file_path = "/opt/conda/lib/python3.8/site-packages/easycv/toolkit/modelscope/pipelines/face_2d_keypoints_pipeline.py"):
    
    line_number = 43 
    code = "det_model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'"
    new_code = "        det_model_id = 'engine/face_2d_keypoints/cv_resnet_facedetection_scrfd10gkps'" 

    modify_line_in_file(file_path, line_number, code, new_code)
