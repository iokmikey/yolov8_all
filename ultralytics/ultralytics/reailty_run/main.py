from ultralytics import YOLO

import torch

# import wandb



def Train_Model(model_path,data,epochs):
    # init
    model = YOLO(model_path)

    # Train

    train_res = model.train(data=data, epochs=epochs)
    print("训练结果",train_res)

    # 验证
    val_res = model.val()
    print("验证",val_res)


    # to onnx   模型保存
    success = model.export(format='onnx')

    success = model.export(format="")

def Detection_Model_Images(model_path,data,name,classes=None): #模型路径   detection数据 保存名字  检测类别过滤
    model = YOLO(model_path)

    det_res = model.predict(source=data,save=True,save_conf=True,save_txt=True,name=name,classes=classes)
    # hide_labels = True, hide_conf = True   不显示label和  conf

    print("预测",det_res)
    # source后为要预测的图片数据集的的路径
    # save=True为保存预测结果
    # save_conf=True为保存坐标信息
    # save_txt=True为保存txt结果，但是yolov8本身当图片中预测不到异物时，不产生txt文件



def Detection_Model_Video(model_path,data,name):
    model = YOLO(model_path)
    res=model.predict(source=data,show=True,save=True,name=name,stream=True)  #stream适用于视频流
    # print(res)
    for i in res:
        print(i.probs)


    # 如果视频则    设置stream=True其余不要 model.predict(source="./det_data"，stream=True）

    # 摄像头 则  0  model.predict(source=0)
    # det_res = model.predict(source=0,show=True,stream=True)
    # for result in det_res:
    #     boxes = result.boxes  #边框输出
    #     masks = result.masks  # 分割掩模输出
    #     keypoints = result.keypoints  # 关键点坐标
    #     probs = result.probs  # 概率  分类  id
    #     print(boxes,masks,keypoints,probs)




#改变图片的后缀，全部变成jpg
def change_image_suffix():
    import os
    import cv2 as cv

    image_path = 'D:/DeskTop/Datasets/clothes/images/'  # 设置图片读取路径
    save_path = 'D:/DeskTop/Datasets/clothes/images_jpg/'  # 设置图片保存路径，新建文件夹，不然其他格式会依然存在

    if not os.path.exists(save_path):  # 判断路径是否正确，并打开
        os.makedirs(save_path)

    image_file = os.listdir(image_path)
    # print(image_file)
    for image in image_file:
        # print(image)
        if image.split('.')[-1] in ['bmp', 'jpg', 'jpeg', 'png', 'JPG', 'PNG']:
            str = image.rsplit(".", 1)  # 从右侧判断是否有符号“.”，并对image的名称做一次分割。如112345.jpeg分割后的str为["112345","jpeg"]
            # print(str)
            output_img_name = str[0] + ".jpg"  # 取列表中的第一个字符串与“.jpg”放在一起。
            # print(output_img_name)
            dir = os.path.join(image_path, image)
            # print("dir:",dir)
            src = cv.imread(dir)
            # print(src)
            cv.imwrite(save_path + output_img_name, src)
    print('FINISHED')


if __name__ == '__main__':


    # pass

    # 填入api   即可查看可视化训练时候图
    # wandb.login(key='08bb4c24e9b0362cd2c8535fdb60a705d36b583a')

    # 加载预训练模型参数

    # model = YOLO('../init_model/yolov8m.pt')


    # Train

    # Train_Model("../init_model/yolov8m.pt", "../datas/coco128.yaml", 3)

    # 初始训练新模型
    # yolo train model = "../init_model/yolov8m.pt" data = "./datas_yaml/SKU-110k.yaml"  epochs = 10  imgsz = 640 batch = 8 --amp = False --workers=0

    # 再次训练
    # yolo train model = "./weights/coco_data/best.pt"  data = "./datas_yaml/coco128.yaml"  epochs = 10  imgsz = 640  batch = 8 resume=True --amp = False --workers=0





    # Detection
    # Detection_Model_Images("./weights/worker_all/best.pt","./det_data/worker_all/video/indianworkers.mp4","rubbish_detection_img")


    # video
    Detection_Model_Video("./weights/face/best.pt",0,"SKU-110K_detection_video")