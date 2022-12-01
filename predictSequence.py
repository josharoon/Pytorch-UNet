
from predict import *
from sequenceHelpers import *

OUTPATH="outputs/images/cadis/Val"
model="checkpoints/checkpoint_epoch200.pth"
net = UNet(n_channels=3, n_classes=8, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Loading model {model}')
logging.info(f'Using device {device}')

net.to(device=device)
net.load_state_dict(torch.load(model, map_location=device))
logging.info('Model loaded!')



imgDirs=getSetDirs(VALIDSET,"Video*/Images")
frames=[]
for d in imgDirs:
    frames = frames+getFrames(d)
    print("predicting Frames for {}".format(d))
    for filename in frames:
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           out_threshold=0.5,
                           device=device,
                           )

        out_filename = Path(OUTPATH).joinpath(Path(filename).parts[-1])
        result = mask_to_image(mask)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')