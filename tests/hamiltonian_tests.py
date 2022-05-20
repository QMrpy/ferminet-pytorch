import torch 

import hamiltonian


def wavefunction_box(x: torch.Tensor):
    """
    Returns the wavefunction of a electron in a 3D box
    Args:
        x: Shape (ndim). Coordinates of the electron
    Returns:
        Wavefunction of the electron in 3D box, with n_i = 1, L_i = 1
        and all constants set to 1, i.e, the wavefunction is,
        psi(X) = sin(x) * sin(y) * sin(z)
    """
    return torch.prod(torch.sin(x))

def test_kinetic_energy_box():
    """
    Tests the correctness of hamiltonian.kinetic() method for 
    elctron in 3D box.
    """

    x = torch.randn(size=(3,))
    psi = wavefunction_box(x)

    actual_kinetic_energy = hamiltonian.kinetic(wavefunction_box, x)
    expected_kinetic_energy = 3 * psi

    torch.testing.assert_close(expected_kinetic_energy, actual_kinetic_energy)

def main():
    test_kinetic_energy_box()

if __name__ == "__main__":
    main()
