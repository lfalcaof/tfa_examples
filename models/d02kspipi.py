"""
Various models to describe D0->KSππ decay
- Simple
- Belle
- BaBar
"""
import uproot
import tensorflow as tf
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm


def simple(x, phsp,
           md, mpi, mkz, 
           mkst, wkst,
           mrho, wrho,
           rd, rr):
    """A simple model in which amplitudes parameters are fixed

    Args:
        x (tf.tensor): the input data
        phsp (DalitzPhaseSpace): Dalitz phase space class to calculate invariant masses and angles
        md (tf.constant): the invariant mass of the D0 candidate
        mpi (tf.constant): the mass of the pion
        mkz (tf.constant): mass of the K0
        mkst (tf.constant): mass of the K*(892) resonance
        wkst (tf.constant): width of the K*(892) resonance
        mrho (tf.constant): mass of the rho(770) resonance
        wrho (tf.constant): width of the rho(770) resonance
        rd (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (D0)
        rr (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (resonance)

    Returns:
        tf.tensor: the value(s) of the amplitude
    """
    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)

    hel_ab = atfd.helicity_amplitude(phsp.cos_helicity_ab(x), 1)
    hel_bc = atfd.helicity_amplitude(phsp.cos_helicity_bc(x), 1)
    hel_ac = atfd.helicity_amplitude(phsp.cos_helicity_ac(x), 1)

    bw1 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw2 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw3 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, mpi, mpi, mkz, md, rd, rr, 1, 1)

    def _model(a1r, a1i, a2r, a2i, a3r, a3i, switches=4 * [1]):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl += a1 * bw1 * hel_ab
        if switches[1]:
            ampl += a2 * bw2 * hel_bc
        if switches[2]:
            ampl += a3 * bw3 * hel_ac
        if switches[3]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        return atfd.density(ampl)

    return _model


def simple_model_mix(x, tphsp, tdz,
           md, mpi, mkz, 
           mkst, wkst,
           mrho, wrho,
           rd, rr):
    """A simple model in which amplitudes parameters are fixed

    Args:
        x (tf.tensor): the input data
        tphsp (CombinedPhaseSpace): Dalitz phase space x Decay times class to calculate invariant masses and angles
        tdz (tf.constant): the lifetime of the D0
        md (tf.constant): the invariant mass of the D0 candidate
        mpi (tf.constant): the mass of the pion
        mkz (tf.constant): mass of the K0
        mkst (tf.constant): mass of the K*(892) resonance
        wkst (tf.constant): width of the K*(892) resonance
        mrho (tf.constant): mass of the rho(770) resonance
        wrho (tf.constant): width of the rho(770) resonance
        rd (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (D0)
        rr (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (resonance)

    Returns:
        tf.tensor: the value(s) of the amplitude
    """

    # DZ - MIXING MODEL
    
    # CACHED VARIABLES
    m2ab = tphsp.phsp1.m2ab(x)
    m2bc = tphsp.phsp1.m2bc(x)
    m2ac = tphsp.phsp1.m2ac(x)

    hel_ab = atfd.helicity_amplitude(tphsp.phsp1.cos_helicity_ab(x), 1)
    hel_bc = atfd.helicity_amplitude(tphsp.phsp1.cos_helicity_bc(x), 1)
    hel_ac = atfd.helicity_amplitude(tphsp.phsp1.cos_helicity_ac(x), 1)

    bw1 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw2 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw3 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, mpi, mpi, mkz, md, rd, rr, 1, 1)

    def _model(a1r, a1i, a2r, a2i, a3r, a3i,
               x_mix_par, y_mix_par, qop_mag, qop_pha, 
               switches=4 * [1]):

        tep = atfm.psip(tphsp.phsp2.t(tphsp.data2(x)), y_mix_par, tdz)
        tem = atfm.psim(tphsp.phsp2.t(tphsp.data2(x)), y_mix_par, tdz)
        tei = atfm.psii(tphsp.phsp2.t(tphsp.data2(x)), x_mix_par, tdz)
        qoverp = atfi.complex(qop_mag * atfi.cos(qop_pha), 
                              qop_mag * atfi.sin(qop_pha))

        #a1r, a1i, a2r, a2i, a3r, a3i = atfi.const(1.0), atfi.const(0.0), atfi.const(0.5), atfi.const(0.0), atfi.const(2.0), atfi.const(0.0)
        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)

        # D0 AMPLITUDE
        ampl_dz = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl_dz += a1 * bw1 * hel_ab
        if switches[1]:
            ampl_dz += a2 * bw2 * hel_bc
        if switches[2]:
            ampl_dz += a3 * bw3 * hel_ac
        if switches[3]:
            ampl_dz += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        # D0bar AMPLITUDE: exchange a <--> c
        ampl_dzb = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl_dzb += a1 * bw2 * hel_bc
        if switches[1]:
            ampl_dzb += a2 * bw1 * hel_ab
        if switches[2]:
            ampl_dzb += -a3 * bw3 * hel_ac
        if switches[3]:
            ampl_dzb += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        # MIXING
        dens = atfm.mixing_density(ampl_dz, ampl_dzb, qoverp, tep, tem, tei)
        return dens

    return _model


def helicity_babar2008(mAB,mBC,mAC,mA,mB,mC,mD,spin):
    hel = atfi.where( spin==1, mAC*mAC-mBC*mBC+((mD*mD-mC*mC)*(mB*mB-mA*mA)/(mAB*mAB)),
                  atfi.where( spin==2, atfi.pow(mBC*mBC-mAC*mAC+(mD*mD-mC*mC)*(mA*mA-mB*mB)/(mAB*mAB),2)-
                                    1./3.*(mAB*mAB-2.*(mD*mD+mC*mC)+atfi.pow(mD*mD-mC*mC,2)/(mAB*mAB))*
                                    (mAB*mAB-2.*(mA*mA+mB*mB)+atfi.pow(mA*mA-mB*mB,2)/(mAB*mAB)),
                             1.0) )    
    return hel


def nonresonant_lass_lineshape_babar2008(m2ab, a, r, ma, mb):
    r"""LASS line shape, nonresonant part

    .. math::

        LASS(m^2) = \frac{m}{q \cot \delta_b - i q}

    
    with :math:`q` is the momentum of the two-body system and :math:`\delta_b` is the scattering phase shift

    .. math::

        \cot \delta_b = \frac{1}{a q} + \frac{1}{2} r q


    from `Aston et al. Nuclear Physics B, Volume 296, Issue 3 (1988), Pages 493-526 <https://doi.org/10.1016/0550-3213(88)90028-4>`_

    Args:
        m2ab (float): invariant mass squared of the system
        a (float): parameter of the effective range term
        r (float): parameter of the effective range term
        ma (float): mass of particle a
        mb (float): mass of particle b

    Returns:
        complex: the nonresonant LASS amplitude
    """
    m = atfi.sqrt(m2ab)
    q = atfk.two_body_momentum(m, ma, mb)
    cot_deltab = 1.0 / a / q + 1.0 / 2.0 * r * q
    ampl = atfi.cast_complex(m) / atfi.complex(q * cot_deltab, -q)
    return ampl

def resonant_lass_lineshape_babar2008(m2ab,
                            m0,
                            gamma0,
                            ma,
                            mb):
    r"""LASS line shape, resonant part

    .. math::

        LASS(m^2) = BW(m^2) (\cos \delta_b + i \sin \delta_b ) ( m_0^2 \Gamma_0 / q_0 )

    Args:
        m2ab (float): invariant mass squared of the system
        m0 (float): resonance mass
        gamma0 (float): resonance width
        a (float): parameter *a* of the effective range term
        r (float): parameter *r* of the effective range term
        md (float): mass of mother particle 
        mc (float): mass of other particle wrt resonance

    Returns:
        complex: the resonant LASS amplitude
    """
    m = atfi.sqrt(m2ab)
    q = atfk.two_body_momentum(m, ma, mb)
    q0 = atfk.two_body_momentum(m0, ma, mb)
    width = gamma0 * q / m * m0 / q0
    ampl = (atfd.relativistic_breit_wigner(m2ab, m0, width) *
            atfi.cast_complex(m0 * m0 * gamma0 / q0))
    return ampl

def LASS_babar2008(m2, a, r, m0, gamma0, ma, mb, amp_res, phase_res, amp_nr, phase_nr):
    m = atfi.sqrt(m2)
    q = atfk.two_body_momentum(m, ma, mb)
    cot_delta_beta = atfi.const(1.0) / a / q + atfi.const(0.5) * r * q
    nr = nonresonant_lass_lineshape_babar2008(m2, a, r, ma, mb)
    res= resonant_lass_lineshape_babar2008(m2, m0, gamma0, ma, mb)
    lass = atfi.cast_complex(amp_res) * \
        atfi.complex( atfi.cos(phase_res + 2.0*phase_nr), atfi.sin(phase_res + 2.0*phase_nr) ) * \
             atfi.complex( q * cot_delta_beta, q ) / atfi.complex( q * cot_delta_beta, -q ) * res
    lass += atfi.cast_complex(amp_nr) * atfi.complex( atfi.cos(phase_nr), atfi.sin(phase_nr) ) * \
           atfi.cast_complex( atfi.cos(phase_nr) + atfi.sin(phase_nr) * cot_delta_beta ) * nr
    return lass


def babar2008_model(x, phsp,
                    mrho, wrho, mkst, wkst,
                    mk2st1430, wk2st1430,
                    mkst1410, wkst1410,
                    mkst1680, wkst1680,
                    momega, womega,
                    mf2_1270, wf2_1270,
                    mrho1450, wrho1450,
                    lass_a, lass_r, lass_M, lass_G, lass_R, lass_phiR, lass_F, lass_phiF,
                    m_poles, g_poles, s0, fij, b_poles, K_matrix_sprod, fprod1,
                    km_masses):

    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)
    mab = atfi.sqrt(m2ab)
    mbc = atfi.sqrt(m2bc)
    mac = atfi.sqrt(m2ac)

    # zemach tensors
    hel_ab_1 = atfi.cast_complex(helicity_babar2008(mab,mbc,mac,phsp.ma,phsp.mb,phsp.mc,phsp.md,1))
    hel_bc_1 = atfi.cast_complex(helicity_babar2008(mbc,mab,mac,phsp.mc,phsp.mb,phsp.ma,phsp.md,1))
    hel_ac_1 = atfi.cast_complex(helicity_babar2008(mac,mbc,mab,phsp.ma,phsp.mc,phsp.mb,phsp.md,1))
    hel_ab_2 = atfi.cast_complex(helicity_babar2008(mab,mbc,mac,phsp.ma,phsp.mb,phsp.mc,phsp.md,2))
    hel_bc_2 = atfi.cast_complex(helicity_babar2008(mbc,mab,mac,phsp.mc,phsp.mb,phsp.ma,phsp.md,2))
    hel_ac_2 = atfi.cast_complex(helicity_babar2008(mac,mbc,mab,phsp.ma,phsp.mc,phsp.mb,phsp.md,2))

    bw1 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw2 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    #bw3 = amplitude_BW_tf(m2ab, wk0st1430, mk0st1430, 1, mpi, mkz, mpi, md)  # using LASS instead
    bw4 = atfd.breit_wigner_lineshape(m2ab, mk2st1430, wk2st1430, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 2, 2, barrier_factor=False)
    bw5 = atfd.breit_wigner_lineshape(m2ab, mkst1410, wkst1410, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw6 = atfd.breit_wigner_lineshape(m2ab, mkst1680, wkst1680, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw7 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    #bw8 = amplitude_BW_tf(m2bc, wk0st1430, mk0st1430, 1, mpi, mkz, mpi, md) # using LASS instead
    bw9 = atfd.breit_wigner_lineshape(m2bc, mk2st1430, wk2st1430, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 2, 2, barrier_factor=False)
    bw10 = atfd.breit_wigner_lineshape(m2bc, mkst1410, wkst1410, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw11 = atfd.breit_wigner_lineshape(m2bc, mkst1680, wkst1680, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw12 = atfd.breit_wigner_lineshape(m2ac, momega, womega, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw13 = atfd.breit_wigner_lineshape(m2ac, mf2_1270, wf2_1270, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 2, 2, barrier_factor=False)
    bw14 = atfd.breit_wigner_lineshape(m2ac, mrho1450, wrho1450, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False, md0=mrho1450 + phsp.mb)

    # LASS
    lass_n = LASS_babar2008(m2ab, lass_a, lass_r, lass_M, lass_G, phsp.mb, phsp.ma, lass_R, lass_phiR, lass_F, lass_phiF)
    lass_p = LASS_babar2008(m2bc, lass_a, lass_r, lass_M, lass_G, phsp.mb, phsp.mc, lass_R, lass_phiR, lass_F, lass_phiF)

    # K matrix
    km = atfd.kmatrix_lineshape(
        m2ac, m_poles, g_poles, s0, fij, b_poles, K_matrix_sprod, fprod1,
        km_masses)

    def _model(a1r,
               a1i,
               a2r,
               a2i,
               a3r,
               a3i,
               a4r,
               a4i,
               a5r,
               a5i,
               a6r,
               a6i,
               a7r,
               a7i,
               a8r,
               a8i,
               a9r,
               a9i,
               a10r,
               a10i,
               a11r,
               a11i,
               a12r,
               a12i,
               a13r,
               a13i,
               a14r,
               a14i,
               switches=15 * [1]+[0]):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)
        a4 = atfi.complex(a4r, a4i)
        a5 = atfi.complex(a5r, a5i)
        a6 = atfi.complex(a6r, a6i)
        a7 = atfi.complex(a7r, a7i)
        a8 = atfi.complex(a8r, a8i)
        a9 = atfi.complex(a9r, a9i)
        a10 = atfi.complex(a10r, a10i)
        a11 = atfi.complex(a11r, a11i)
        a12 = atfi.complex(a12r, a12i)
        a13 = atfi.complex(a13r, a13i)
        a14 = atfi.complex(a14r, a14i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0))

        if switches[0]:
            ampl += a1 * bw1 * hel_ac_1
        if switches[1]:
            ampl += a2 * bw2 * hel_ab_1
        if switches[2]:
            ampl += a3 * lass_n
        if switches[3]:
            ampl += a4 * bw4 * hel_ab_2
        if switches[4]:
            ampl += a5 * bw5 * hel_ab_1
        if switches[5]:
            ampl += a6 * bw6 * hel_ab_1
        if switches[6]:
            ampl += a7 * bw7 * hel_bc_1
        if switches[7]:
            ampl += a8 * lass_p
        if switches[8]:
            ampl += a9 * bw9 * hel_bc_2
        if switches[9]:
            ampl += a10 * bw10 * hel_bc_1
        if switches[10]:
            ampl += a11 * bw11 * hel_bc_1
        if switches[11]:
            ampl += a12 * bw12 * hel_ac_1
        if switches[12]:
            ampl += a13 * bw13 * hel_ac_2
        if switches[13]:
            ampl += a14 * bw14 * hel_ac_1
        if switches[14]:
            ampl += km
        if switches[15]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0))

        return atfd.density(ampl)

    return _model



def babar2008_model_amp(x, phsp,
                    mrho, wrho, mkst, wkst,
                    mk2st1430, wk2st1430,
                    mkst1410, wkst1410,
                    mkst1680, wkst1680,
                    momega, womega,
                    mf2_1270, wf2_1270,
                    mrho1450, wrho1450,
                    lass_a, lass_r, lass_M, lass_G, lass_R, lass_phiR, lass_F, lass_phiF,
                    m_poles, g_poles, s0, fij, b_poles, K_matrix_sprod, fprod1,
                    km_masses):

    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)
    mab = atfi.sqrt(m2ab)
    mbc = atfi.sqrt(m2bc)
    mac = atfi.sqrt(m2ac)

    # zemach tensors
    hel_ab_1 = atfi.cast_complex(helicity_babar2008(mab,mbc,mac,phsp.ma,phsp.mb,phsp.mc,phsp.md,1))
    hel_bc_1 = atfi.cast_complex(helicity_babar2008(mbc,mab,mac,phsp.mc,phsp.mb,phsp.ma,phsp.md,1))
    hel_ac_1 = atfi.cast_complex(helicity_babar2008(mac,mbc,mab,phsp.ma,phsp.mc,phsp.mb,phsp.md,1))
    hel_ab_2 = atfi.cast_complex(helicity_babar2008(mab,mbc,mac,phsp.ma,phsp.mb,phsp.mc,phsp.md,2))
    hel_bc_2 = atfi.cast_complex(helicity_babar2008(mbc,mab,mac,phsp.mc,phsp.mb,phsp.ma,phsp.md,2))
    hel_ac_2 = atfi.cast_complex(helicity_babar2008(mac,mbc,mab,phsp.ma,phsp.mc,phsp.mb,phsp.md,2))

    bw1 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw2 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    #bw3 = amplitude_BW_tf(m2ab, wk0st1430, mk0st1430, 1, mpi, mkz, mpi, md)  # using LASS instead
    bw4 = atfd.breit_wigner_lineshape(m2ab, mk2st1430, wk2st1430, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 2, 2, barrier_factor=False)
    bw5 = atfd.breit_wigner_lineshape(m2ab, mkst1410, wkst1410, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw6 = atfd.breit_wigner_lineshape(m2ab, mkst1680, wkst1680, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw7 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    #bw8 = amplitude_BW_tf(m2bc, wk0st1430, mk0st1430, 1, mpi, mkz, mpi, md) # using LASS instead
    bw9 = atfd.breit_wigner_lineshape(m2bc, mk2st1430, wk2st1430, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 2, 2, barrier_factor=False)
    bw10 = atfd.breit_wigner_lineshape(m2bc, mkst1410, wkst1410, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw11 = atfd.breit_wigner_lineshape(m2bc, mkst1680, wkst1680, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw12 = atfd.breit_wigner_lineshape(m2ac, momega, womega, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False)
    bw13 = atfd.breit_wigner_lineshape(m2ac, mf2_1270, wf2_1270, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 2, 2, barrier_factor=False)
    bw14 = atfd.breit_wigner_lineshape(m2ac, mrho1450, wrho1450, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1.5, 5.0, 1, 1, barrier_factor=False, md0=mrho1450 + phsp.mb)

    # LASS
    lass_n = LASS_babar2008(m2ab, lass_a, lass_r, lass_M, lass_G, phsp.mb, phsp.ma, lass_R, lass_phiR, lass_F, lass_phiF)
    lass_p = LASS_babar2008(m2bc, lass_a, lass_r, lass_M, lass_G, phsp.mb, phsp.mc, lass_R, lass_phiR, lass_F, lass_phiF)

    # K matrix
    km = atfd.kmatrix_lineshape(
        m2ac, m_poles, g_poles, s0, fij, b_poles, K_matrix_sprod, fprod1,
        km_masses)

    def _model(a1r,
               a1i,
               a2r,
               a2i,
               a3r,
               a3i,
               a4r,
               a4i,
               a5r,
               a5i,
               a6r,
               a6i,
               a7r,
               a7i,
               a8r,
               a8i,
               a9r,
               a9i,
               a10r,
               a10i,
               a11r,
               a11i,
               a12r,
               a12i,
               a13r,
               a13i,
               a14r,
               a14i,
               switches=15 * [1]+[0]):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)
        a4 = atfi.complex(a4r, a4i)
        a5 = atfi.complex(a5r, a5i)
        a6 = atfi.complex(a6r, a6i)
        a7 = atfi.complex(a7r, a7i)
        a8 = atfi.complex(a8r, a8i)
        a9 = atfi.complex(a9r, a9i)
        a10 = atfi.complex(a10r, a10i)
        a11 = atfi.complex(a11r, a11i)
        a12 = atfi.complex(a12r, a12i)
        a13 = atfi.complex(a13r, a13i)
        a14 = atfi.complex(a14r, a14i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0))

        if switches[0]:
            ampl += a1 * bw1 * hel_ac_1
        if switches[1]:
            ampl += a2 * bw2 * hel_ab_1
        if switches[2]:
            ampl += a3 * lass_n
        if switches[3]:
            ampl += a4 * bw4 * hel_ab_2
        if switches[4]:
            ampl += a5 * bw5 * hel_ab_1
        if switches[5]:
            ampl += a6 * bw6 * hel_ab_1
        if switches[6]:
            ampl += a7 * bw7 * hel_bc_1
        if switches[7]:
            ampl += a8 * lass_p
        if switches[8]:
            ampl += a9 * bw9 * hel_bc_2
        if switches[9]:
            ampl += a10 * bw10 * hel_bc_1
        if switches[10]:
            ampl += a11 * bw11 * hel_bc_1
        if switches[11]:
            ampl += a12 * bw12 * hel_ac_1
        if switches[12]:
            ampl += a13 * bw13 * hel_ac_2
        if switches[13]:
            ampl += a14 * bw14 * hel_ac_1
        if switches[14]:
            ampl += km
        if switches[15]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0))

        return ampl

    return _model


def belle_cache_amp(x, phsp, file_name):
    # Cached information
    cached_model = uproot.open(file_name)
    phase_values, binning_x, binning_y = cached_model["histo_phase"].to_numpy()
    ampl_values, _, _ = cached_model["histo_amplitude"].to_numpy()
    # calculate phase space variables for the input data
    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)
    # get index of the bin for each event
    bin_x = tf.cast(tf.floor((m2ab - binning_x[0]) / (binning_x[1] - binning_x[0])), tf.int32)
    bin_y = tf.cast(tf.floor((m2bc - binning_y[0]) / (binning_y[1] - binning_y[0])), tf.int32)
    # get phase and amplitude values for each event
    phase = tf.gather_nd(phase_values, tf.stack([bin_x, bin_y], axis=-1))
    ampl_mod = tf.gather_nd(ampl_values, tf.stack([bin_x, bin_y], axis=-1))
    ampl = atfi.cast_complex(atfi.complex_from_polar(ampl_mod, phase))
    return ampl