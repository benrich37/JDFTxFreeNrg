Includes implementation of the quasi-rigid rotor harmonic oscillator (QRRHO) approach for vibrational entropies (Grimme, 2012) and enthalpies (Li, Head-Gordon, 2014).

# Outline

## Solvation entropies

Implementation of equations 2-12, 45 and 46 from Solvation Entropy Made Simple (Garza 2019). 
This includes a suitable term for volume in the translational entropy given by an effective cavity volume, adjustments to the translational entropy arising from the loss of free space from a species' rotation, and a special case adjustment to the rotational partition function for a restricted rotor.
This also includes a `StructureVolume(Structure)` class, which uses Monte Carlo integration for estimating Van der Waals structure volumes needed for these entropies.
The analytical equation for the union volume of two spheres from Van der Waals Volumes and Radii (Bondi 1963) is also implemented in the testing directory for benchmarking Monte Carlo volumes.

## Vibrational Entropies

Implements equations 3, 4, and 6-8 from Supramolecular Binding Thermodynamics by Dispersion-Corrected Density Functional Theory (Grimme 2012).
This includes a quasi rigid-rotor description for the entropy of a vibrational mode as a rotation of matching frequency, the standard harmonic oscillator description for a mode's entropy, and a mixing function to scale between the two about a critical frequency (reccomended 100 cm^-1).

## Vibrational Enthalpies

Implements equations 4-6 from Improved Force-Field Parameters for QM/MM Simulations of the Energies of Adsorption for Molecules in Zeolites and a Free Rotor Correction to the Rigid Rotor Harmonic Oscillator Model for Adsorption Enthalpies (Li, Head-Gordon 2014).
This includes the enthalpy of a single free rotational degree of freedom, the standard harmonic oscillator enthalpy of a vibrational mode, and a mixing function to scale between the two (identical to one used for entropy).

## Vibrational Frequencies

Includes functions to read the hessian dumped in a JDFTx vibrational analysis calculation, project out specified rotations and translation of subspecies from the computed system, and extract vibrational frequencies.

