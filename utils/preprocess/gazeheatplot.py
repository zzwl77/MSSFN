import os
import argparse
import csv
import numpy
import matplotlib
from matplotlib import pyplot, image

def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = dispsize[0] / 2 - w / 2
        y = dispsize[1] / 2 - h / 2
        y = int(round(y))
        x = int(round(x))
        h = int(round(h))
        w = int(round(w))
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig, ax

def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M

def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y)
    
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = gwh // 2
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)
        pyplot.close(fig)  # CRITICAL: Close figure to free memory
        return None  # No need to return figure when saved/closed
    else:
        return fig

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

def gaussian(size, sigma):
    """Generate a Gaussian kernel."""
    x = np.arange(size) - size // 2
    y = x[:, None]
    x0, y0 = 0, 0  # Center
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))

def draw_heatmap2(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """
    使用注视点生成热图，并将热图作为权重直接作用在背景图上。

    参数:
    gazepoints    -- 注视点数据，包含每个注视点的 (x, y, duration) 信息。
    dispsize      -- 显示的尺寸，例如 (1920, 1080)。
    imagefile     -- 背景图像文件路径，将被热图加权。
    alpha         -- 控制权重的整体强度，值越高，注视点的影响越强。
    savefilename  -- 保存生成的图像路径；如果为 None，则不保存图像。
    gaussianwh    -- 高斯核的宽度。
    gaussiansd    -- 高斯核的标准差。
    
    返回:
    fig           -- 包含生成图像的 Matplotlib Figure 实例。
    """

    # 加载背景图像

    # 创建热图
    gwh = gaussianwh
    gsdwh = gwh / 6 if gaussiansd is None else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    strt = gwh // 2
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = np.zeros(heatmapsize, dtype=float)
    
    # 对每个注视点将高斯核叠加到热图上
    for i in range(len(gazepoints)):
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        
        # 修正高斯核大小以防止超出显示边界
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])

            # 将修正后的高斯核叠加到热图上
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                pass  # 注视点可能在显示范围外
        else:
            # 将高斯核叠加到热图上
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]

    # 调整热图大小到原始显示尺寸
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # # 归一化热图，将非空值缩放到 [0, 1] 之间，空值为0
    heatmap = np.nan_to_num(heatmap / np.nanmax(heatmap))

    # 加载并调整背景图像
    background_img1 = Image.open(imagefile).convert("RGB")
    background_img = np.array(background_img1.resize(dispsize))
    # background_img1.close()  # 转换后立即释放 PIL 图像内存

    # 将热图扩展为三通道以匹配 RGB 格式
    heatmap_rgb = np.dstack([heatmap] * 3)

    # 将热图作为权重直接应用到背景图上
    weighted_img = background_img * heatmap_rgb
    weighted_img = np.clip(weighted_img, 0, 255).astype(np.uint8)  # 确保像素值在 [0, 255]
    weighted_image = Image.fromarray(weighted_img)
    if savefilename:
        weighted_image.save(savefilename)

    return weighted_image  # 返回 PIL 图像对象，以便进一步处理或释放内存
##################
#     Parsing    #
##################

# parser = argparse.ArgumentParser(description='Parameters required for processing.')

# #required args
# parser.add_argument('input-path', type=str, help='path to the csv input')
# parser.add_argument('display-width', type=int, help='an integer representing the display width')
# parser.add_argument('display-height', type=int, help='an integer representing the display height')

# #optional args
# parser.add_argument('-a', '--alpha', type=float, default='0.5', required=False, help='alpha for the gaze overlay')
# parser.add_argument('-o',  '--output-name', type=str, required=False, help='name for the output file')
# parser.add_argument('-b',  '--background-image', type=str, default=None, required=False, help='path to the background image')

# #advanced optional args
# parser.add_argument('-n', '--n-gaussian-matrix', type=int, default='200', required=False, help='width and height of gaussian matrix')
# parser.add_argument('-sd',  '--standard-deviation', type=float, default=None ,required=False, help='standard deviation of gaussian distribution')


# args = vars(parser.parse_args())

# input_path = args['input-path']
# display_width = args['display-width']
# display_height = args['display-height']
# alpha = args['alpha']
# output_name = args['output_name'] if args['output_name'] is not None else 'output'
# background_image = args['background_image']
# ngaussian = args['n_gaussian_matrix']
# sd = args['standard_deviation']

# with open(input_path) as f:
# 	reader = csv.reader(f)
# 	raw = list(reader)
	
# 	gaza_data = []
# 	if len(raw[0]) is 2:
# 		gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw))
# 	else:
# 		gaze_data =  list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw))
		
# 	draw_heatmap(gaze_data, (display_width, display_height), alpha=alpha, savefilename=output_name, imagefile=background_image, gaussianwh=ngaussian, gaussiansd=sd)