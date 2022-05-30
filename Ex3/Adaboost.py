import numpy as np

NUM_POINTS = 150
S_SIZE = 75
NUM_ITER = 8
NUM_ROUNDS = 100


def calc_line_eq_params(p1: tuple, p2: tuple):
    """
        calculate line equation params based on two points
    """
    x1, x2, y1, y2 = p1[0], p2[0], p1[1], p2[1]
    if np.abs(x1 - x2) < 0.0001:
        m = None
        n = x1
    else:
        m = (y1 - y2) / (x1 - x2)
        n = y1 - (m * x1)
    return m, n


def get_vector_point_labeling(m: float, n: float, points: np.ndarray, direction: int) -> np.ndarray:
    """
        set prediction value (-1,1) to points vector,
        for every point above line label with 1,else -1
    """
    p_x, p_y, = points[:, 0].reshape(1, -1), points[:, 1].reshape(1, -1)
    points_values = np.zeros((1, S_SIZE))
    if m is not None:
        p_x_line_projection = p_x * m + n
        points_values[p_x_line_projection >= p_y] = 1 * direction
        points_values[p_x_line_projection < p_y] = -1 * direction
    else:
        points_values[p_x >= n] = 1 * direction
        points_values[p_x < n] = -1 * direction

    return points_values


def is_scalar_point_above_line(m: float, n: float, p: tuple, direction: int) -> int:
    """
        set prediction value (-1,1) to single point,
        if point above line label with 1,else -1
    """
    p_x, p_y = p[0], p[1]
    if m is not None:
        p_x_on_line_projection = p_x * m + n
        if p_x_on_line_projection < p_y:
            return -1 * direction  # above the line
        else:
            return 1 * direction  # under the line
    else:
        if p_x >= n:
            return 1 * direction
        else:
            return -1 * direction


def model_eval(best_rules_list: list,
               rules_weight: np.ndarray,
               test_features: np.ndarray,
               gt_pred: np.ndarray,
               s_x: np.ndarray,
               hyp_permutations: np.ndarray,
               directions: np.ndarray) -> float:
    """
        calculate model error
    """
    pred = np.zeros_like(gt_pred)
    rules_error = np.zeros(NUM_ITER)

    for idx, rule_idx in enumerate(best_rules_list):
        rules_list = best_rules_list[:idx + 1]
        for index, point in enumerate(test_features):
            error = 0
            for i_rule in rules_list:
                rule = hyp_permutations[i_rule // 2]
                p_1, p_2 = s_x[rule[0]], s_x[rule[1]]
                m, n = calc_line_eq_params(p_1, p_2)
                error += rules_weight[i_rule] * is_scalar_point_above_line(m, n, point, directions[i_rule])
            pred[index] = 1 if error > 0 else -1
        rules_error[idx] = (pred != gt_pred).sum() / S_SIZE

    return rules_error











