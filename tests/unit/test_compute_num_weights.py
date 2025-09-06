import numpy as np

import elicito as el


def test_compute_num_weights():
    sum = el.utils.compute_num_weights([(3, 4), (4, 5), (5,), (5, None)])
    expected_sum = 3 * 4 + 4 * 5 + 5 + 5

    np.testing.assert_equal(sum, expected_sum)
