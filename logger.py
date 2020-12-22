import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        #img_summaries = []
        # for i, img in enumerate(images):
        # Write the image to a string
        # try:
        # s = StringIO()
        # except:
        # s = BytesIO()
        # scipy.misc.toimage(img).save(s, format="png")
        # img = Image.fromarray(np.uint16(img), mode='L')
        # img.save(s, format("png"))
        # print(img.shape)
        # Image.fromarray((img[:, :, 0] * 255).astype('uint8'), mode='L').convert('RGB').save(s, format="png")

        # Create an Image object
        # img_sum = tf.summary.image(encoded_image_string=s.getvalue(),
        # height=img.shape[0],
        # width=img.shape[1])
        # Create a Summary value
        # img_summaries.append(tf.summary.value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        # summary = tf.summary(value=img_summaries)
        # self.writer.add_summary(summary, step)
        # img_grid = torchvision.utils.make_grid(images)
        # self.writer.add_image(tag, images, step)
        # self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        # counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        '''
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.summary(value=[tf.summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        '''
        tf.summary.histogram(name=tag, data=values, step=step)
        self.writer.flush()

    def close_summary(self):
        self.writer.close()
