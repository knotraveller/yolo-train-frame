# python code/main.py --train --val --predict -d "Car Data" -m yolo12n.pt
# python code/main.py --train -d "Car Data" -m 'yolo12n.pt' -e 1
python code/main.py --predict -d "armor_dataset(part)/images" -m './model/armor.onnx'
# python code/main.py -m "./runs/Kfolder/train0/weights/best.pt" -d "splited_data/fold0/val/images" --predict