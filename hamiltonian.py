import torch
import torch.autograd.functional as functional


def kinetic(psi, position: torch.Tensor) -> torch.Tensor:
	"""
	Returns the kinetic energy in the form of -(nabla^2 f)
	Args:
		psi: Python function. The electron wavefunction
		positions: Shape (ndim). Coordinates of the electron
	Returns:
		Kinetic energy of the electron at the given position
	"""

	return -torch.trace(functional.hessian(psi, position))
	
def potential_electron_electron(r_ee: torch.Tensor) -> torch.Tensor:
	"""
	Returns the electron-electron potential.
	Args:
		r_ee: Shape (nelectrons, nelectrons, :). r_ee[i, j, 0] gives the distance
		between electrons i and j. Other elements in the final axes are not
		required.
	"""

	return torch.sum(torch.triu(r_ee[..., 0], diagonal=1))

def potential_electron_nuclear(charges: torch.Tensor, 
    r_ne: torch.Tensor) -> torch.Tensor:
	"""
	Returns the electron-nuclear potential.
	Args:
		charges: Shape (natoms). Nuclear charges of the atoms.
		r_ne: Shape (nelectrons, natoms). r_ne[i, j] gives the distance between
		electron i and atom j.
	"""

	return -1 * torch.sum(charges / r_ne[..., 0])

def potential_nuclear_nuclear(charges: torch.Tensor,
    atoms: torch.Tensor) -> torch.Tensor:
	"""
	Returns the nuclear-nuclear potential.
	Args:
		charges: Shape (natoms). Nuclear charges of the atoms.
		atoms: Shape (natoms, ndim). Positions of the atoms.
	"""

	r_nn = torch.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
	return torch.sum(torch.triu((charges[None, ...] * charges[..., None]) / r_nn, 
		diagonal=1))

def potential_energy(r_ne: torch.Tensor, r_ee: torch.Tensor, atoms: torch.Tensor,
    charges: torch.Tensor) -> torch.Tensor:
	"""
	Returns the potential energy for this electron configuration.
	Args:
		r_ne: Shape (nelectrons, natoms). r_ne[i, j] gives the distance between
		electron i and atom j.
		r_ee: Shape (nelectrons, nelectrons, :). r_ee[i, j, 0] gives the distance
		between electrons i and j. Other elements in the final axes are not
		required.
		atoms: Shape (natoms, ndim). Positions of the atoms.
		charges: Shape (natoms). Nuclear charges of the atoms.
	"""

	return (potential_electron_electron(r_ee) +
		potential_electron_nuclear(charges, r_ne) +
		potential_nuclear_nuclear(charges, atoms))