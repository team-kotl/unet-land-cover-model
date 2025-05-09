import segmentation_models_pytorch as smp

def create_unet():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=5,
        activation=None,
    )