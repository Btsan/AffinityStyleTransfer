import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision import io

from pathlib import Path

from torch import nn
import torch.nn.functional as F
import torchvision.models as models

from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGEncoder(nn.Module):
	def __init__(self, path=None):
		super(VGGEncoder, self).__init__()

		if path is not None:
			vgg = vgg_normalised = nn.Sequential( # Sequential,
				nn.Conv2d(3,3,(1, 1)),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(3,64,(3, 3)),
				nn.ReLU(), # 1 1
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(64,64,(3, 3)),
				nn.ReLU(),
				nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(64,128,(3, 3)),
				nn.ReLU(), # 2 1
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(128,128,(3, 3)),
				nn.ReLU(),
				nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(128,256,(3, 3)),
				nn.ReLU(), # 3 1
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,256,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,256,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,256,(3, 3)),
				nn.ReLU(),
				nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,512,(3, 3)),
				nn.ReLU(), # 4 1
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,512,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,512,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,512,(3, 3)),
				nn.ReLU(),
				nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,512,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,512,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,512,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,512,(3, 3)),
				nn.ReLU(),
			)
			save_path = Path(path)
			if save_path.is_file():
				try:
					state_dict = torch.load(str(save_path))
					vgg.load_state_dict(state_dict)
					print("Loaded vgg", save_path)
				except:
					print("Error: previous vgg failed to load")
					pass
			else:
				vgg = models.vgg19(pretrained=True).features.eval()
				print("Initializing new vgg")
		else:
			vgg = models.vgg19(pretrained=True).features.eval()
		features = list(vgg.children())
		self.enc_1 = nn.Sequential(*features[0:4]) 
		self.enc_2 = nn.Sequential(*features[4:11])
		self.enc_3 = nn.Sequential(*features[11:18])
		self.enc_4 = nn.Sequential(*features[18:31]) # content layer
		self.model = nn.Sequential(*features[0:30])

	def forward(self, x):
		x = self.model(x)
		self.content = F.relu(x).detach() # save content map for loss
		return x

	def compute_loss(self, g, s):
		g1 = self.enc_1(g)
		g2 = self.enc_2(g1)
		g3 = self.enc_3(g2)
		g4 = self.enc_4(g3)
		content_loss = F.mse_loss(g4, self.content)

		if s.dim() == 3:
			s = s.unsqueeze(0)

		s1 = self.enc_1(s)
		s2 = self.enc_2(s1)
		s3 = self.enc_3(s2)
		s4 = self.enc_4(s3)
		style_loss = F.mse_loss(g1.mean((2, 3)), s1.mean((2, 3))) \
			+ F.mse_loss(g2.mean((2, 3)), s2.mean((2, 3))) \
			+ F.mse_loss(g3.mean((2, 3)), s3.mean((2, 3))) \
			+ F.mse_loss(g4.mean((2, 3)), s4.mean((2, 3)))
		style_loss += F.mse_loss(g1.std((2, 3)), s1.std((2, 3))) \
			+ F.mse_loss(g2.std((2, 3)), s2.std((2, 3))) \
			+ F.mse_loss(g3.std((2, 3)), s3.std((2, 3))) \
			+ F.mse_loss(g4.std((2, 3)), s4.std((2, 3)))

		return content_loss, style_loss

class AdaIN(nn.Module):
	def __init__(self, encoder=None):
		super(AdaIN, self).__init__()

		if encoder is None:
			self.model = nn.Sequential(
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(512,256,(3, 3)),
				nn.ReLU(),
				nn.UpsamplingNearest2d(scale_factor=2),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,256,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,256,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,256,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(256,128,(3, 3)),
				nn.ReLU(),
				nn.UpsamplingNearest2d(scale_factor=2),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(128,128,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(128,64,(3, 3)),
				nn.ReLU(),
				nn.UpsamplingNearest2d(scale_factor=2),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(64,64,(3, 3)),
				nn.ReLU(),
				nn.ReflectionPad2d((1, 1, 1, 1)),
				nn.Conv2d(64,3,(3, 3)))
		else:
			children = [] 
			layers = list(encoder.children())
			for i in range(len(layers)-1, -1, -1):
				if isinstance(layers[i], nn.Conv2d):
					in_channels, out_channels = layers[i].weight.shape[:2]
					print("conv_{}".format(i), in_channels, out_channels)
					children.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect'))
					children.append(nn.ReLU())
				elif isinstance(layers[i], nn.MaxPool2d):
					children.append(nn.Upsample(scale_factor=2, mode='nearest'))
			children.pop()
			print(children)
			self.model = nn.Sequential(*children)

	def forward(self, content, style1, style2=None, alpha=1.0):
		target_std = style1.std((2, 3))
		target_mean = style1.mean((2, 3))
		adain = [F.instance_norm(content[i].unsqueeze(0), weight=target_std[i], bias=target_mean[i]) for i  in range(len(content))]
		adain = torch.cat(adain)
		if style2 is not None and alpha < 1:
			target_std = style2.std((2, 3))
			target_mean = style2.mean((2, 3))
			adain2 = [F.instance_norm(content[i].unsqueeze(0), weight=target_std[i], bias=target_mean[i]) for i  in range(len(content))]
			adain2 = torch.cat(adain2)
			adain = (1 - alpha) * adain2 + alpha * adain
		elif alpha < 1:
			adain = (1 - alpha) * content + alpha * adain
		adain = F.relu(adain) # assumes last vgg layer was conv2d
		y = self.model(adain)
		y = torch.tanh(y)
		return y
		# 1 51 95 206 233
class ImageFolderDataset(data.Dataset):
	def __init__(self, folder, transform=None):
		super(ImageFolderDataset, self).__init__()
		self.folder = folder
		self.images = list(Path(self.folder).glob('**/*.*'))
		self.images.sort()
		self.transform = transform

	# i^th item from the dataset
	def __getitem__(self, index):
		index %= len(self.images)
		img = self.images[index]
		try:
			img = io.read_image(str(img))
		except Exception as e:
			# print("Corrupted: " + str(self.images[index]))
			self.images.pop(index)
			return self[index % len(self.images)]
		if self.transform is not None:
			img = self.transform(img)
		if img.shape[0] == 1:
			img = img.expand((3, -1, -1))
		elif img.shape[0] == 4:
			img =  img[:3, :, :]
		return img

	def __len__(self):
		return len(self.images)

	def name(self):
		return 'ImageFolderDataset'

def style_interp(ft1, ft2, alpha=0.5):
	return ft1 * (1.0 - alpha) + ft2 * alpha 

def test(encoder, decoder, name="test.mp4", imsize=256, davis=True, video_path='motorbike', style1="styles/vangogh_starry_night.jpg", style2="styles/10313.jpg"):
	resize = transforms.Resize((imsize, imsize))
	if davis:
		dataset = ImageFolderDataset("datasets/DAVIS/JPEGImages/480p/" + video_path) #,resize
		dataloader = data.DataLoader(dataset, 1, shuffle=False)
	else:
		video = io.read_video(video_path, pts_unit='sec')[0]
		dataloader = video.permute(0, 3, 1, 2)

	pred = []
	with torch.no_grad():
		style_im = io.read_image(style1).unsqueeze(0).to(device) / 255
		features_1 = encoder(resize(style_im))
		style_im = io.read_image(style2).unsqueeze(0).to(device) / 255
		features_2 = encoder(resize(style_im))
		del style_im
		for idx, im in enumerate(dataloader):
			if im.dim() < 4:
				im = im.unsqueeze(0) 
			im = im.to(device) / 255
			p = decoder(encoder(im), features_1, style2=features_2, alpha=1-idx/len(dataloader))
			pred.append(p.cpu())

		if len(pred[0].shape) == 4:
				pred = torch.cat(pred, dim=0)
		else:
			pred = torch.stack(pred, dim=0)
		pred = pred.permute(0, 2, 3, 1)
		pred = (pred * 255 + 0.5).clamp(0, 255)
		io.write_video(name, pred, 10)
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='AdaIN')
	parser.add_argument('--video', type=str, metavar='<dir or .mp4>', default='motorbike', help='path to video folder or mp4 (default=\'DAVIS/motorbike\')')
	parser.add_argument('--davis', type=int, metavar='0 or 1', default=1, help='use davis dataset or not (default=1')
	parser.add_argument('--style1', type=str, metavar='<.jpg>', default="styles/vangogh_starry_night.jpg", help='style image to use (default="styles/vangogh_starry_night.jpg"')
	parser.add_argument('--style2', type=str, metavar='<.jpg>', default="styles/10313.jpg", help='style image to use (default="styles/10313.jpg"')
	parser.add_argument('--decoder', type=str, metavar='<.pth>', default="saves/decoder.pth", help='decoder weights to use (default="saves/decoder.pth"')
	parser.add_argument('--save', type=str, metavar='<.mp4>', default="test.mp4", help='save results as (default="test.mp4"')
	args = parser.parse_args()

	encoder = VGGEncoder("saves/vgg_normalised.pth").to(device)
	decoder = AdaIN().to(device)
	save_path = Path(args.decoder)
	if save_path.is_file():
		try:
			state_dict = torch.load(str(save_path))
			decoder.model.load_state_dict(state_dict)
			print("Loaded previous decoder", save_path)
		except:
			print("Error: previous decoder failed to load")
			pass
	else:
		print("No previous save. Initializing new decoder")

	test(encoder, decoder, name=args.save, davis=bool(args.davis), video_path=args.video, style1=args.style1, style2=args.style2)