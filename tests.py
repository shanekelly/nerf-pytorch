import unittest

import torch

from run_nerf_helpers import get_section_n_samples


class TestStringMethods(unittest.TestCase):

    def test_get_section_n_samples(self):
        """
        @brief - Check that the get_section_n_samples function returns a number of samples that are
            close to the theoretically expected number of samples.
        """
        n_train_imgs = 5
        grid_size = 8

        # Define a random probability distribution over all image sections.
        prob_dist = torch.rand((n_train_imgs, grid_size, grid_size))
        prob_dist /= torch.sum(prob_dist)  # Ensure it sums to 1.
        # Define the number of samples to draw from the distribution.
        n_samples = int(1e6)
        # Draw from the distribution.
        section_n_samples, _ = get_section_n_samples(prob_dist, n_samples)

        # Check that the actual number of times each element was drawn does not differ too much from
        # the theoretical expected number of times.
        thresh = 500
        diff = prob_dist * n_samples - section_n_samples.float()
        self.assertTrue(torch.max(torch.abs(diff)) <= thresh)


if __name__ == '__main__':
    unittest.main()
