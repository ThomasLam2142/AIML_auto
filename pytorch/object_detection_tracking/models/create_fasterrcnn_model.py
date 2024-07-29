from fasterrcnn_resnet50_fpn_v2 import create_model as create_fasterrcnn_resnet50_fpn_v2

def return_fasterrcnn_resnet50_fpn(num_classes, pretrained=True, coco_model=False):
    raise NotImplementedError("This model is not used in the simplified script.")

def return_fasterrcnn_resnet50_fpn_v2(num_classes, pretrained=True, coco_model=False):
    return create_fasterrcnn_resnet50_fpn_v2(num_classes, pretrained, coco_model)

create_model = {
    'fasterrcnn_resnet50_fpn_v2': return_fasterrcnn_resnet50_fpn_v2
}