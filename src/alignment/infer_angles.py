import numpy as np

def infer_angles(decryption_angles, intensity_measurements):
    """
    Infer the original angles using the algorithm described in the Fast Phase Inference.
    
    :param decryption_angles: List of lists of decryption angles used in measurements
    :param intensity_measurements: List of corresponding intensity measurements
    :return: Inferred original angles
    """
    vec = np.zeros(len(decryption_angles[0])**2, dtype=complex)
    mat = np.zeros((len(vec), len(vec)), dtype=complex)
    for angle_set, intensity in zip(decryption_angles, intensity_measurements):
        phase_vec = np.exp(1j * np.array(angle_set))
        b_mat = np.outer(phase_vec, np.conj(phase_vec))
        b_vec = b_mat.flatten()
        vec += intensity * b_vec
        mat += np.outer(b_vec, b_vec)
    a_vec = np.linalg.pinv(mat) @ vec
    return np.angle(a_vec[:len(decryption_angles[0])])