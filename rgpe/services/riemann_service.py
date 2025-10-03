import mpmath as mp


def get_Z_function_terms_features(t: float, max_term: int = 10):
    """
    Obtém os termos de 2 até 10 da função Z
    """
    theta = mp.siegeltheta(t)
    features = {}
    for n in range(2, max_term+1):
        angle = theta - t * mp.ln(n)
        term = 2 * mp.cos(angle) / mp.sqrt(n)
        features[f"z_term_{n}"] = term
    return features


def get_gram_point(n: int, t0: float) -> float:
    """
    Calcula o n-ésimo Gram point usando t0 como chute inicial.
    θ(t) = nπ
    """
    f = lambda t: mp.siegeltheta(t) - n * mp.pi
    return mp.findroot(f, t0)


def get_cogram_point(n: int, t0: float) -> float:
    """Resolve θ(t) = nπ + π/2 perto de t0 (Gram point g_n usado como chute)."""
    f = lambda t: mp.siegeltheta(t) - (n + 0.5) * mp.pi
    return mp.findroot(f, t0)
