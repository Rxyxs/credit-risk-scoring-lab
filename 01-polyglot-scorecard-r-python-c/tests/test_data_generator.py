from src.data_generator import generate_applicants


def test_generate_applicants_shape_and_columns():
    df = generate_applicants(n=2000, seed=1)
    assert len(df) == 2000
    expected_cols = {
        "applicant_id", "edad", "region", "tipo_contrato", "renta_liquida",
        "antiguedad_laboral_meses", "n_productos_activos", "deuda_total",
        "dti", "n_morosidad_reportes", "default_12m",
    }
    assert expected_cols.issubset(df.columns)


def test_default_rate_is_realistic():
    """La tasa de default de cartera debe quedar en un rango plausible
    para credito de consumo/retail chileno (no colapsar a ~0% ni disparar
    a ~20%+, que serian senales de una mala calibracion del proceso
    generador -- ver la recalibracion documentada en el README)."""
    df = generate_applicants(n=20_000, seed=42)
    rate = df["default_12m"].mean()
    assert 0.03 < rate < 0.12


def test_informal_segment_is_riskier_than_formal():
    df = generate_applicants(n=20_000, seed=42)
    rates = df.groupby("tipo_contrato")["default_12m"].mean()
    assert rates["informal"] > rates["formal"]


def test_morosidad_reports_gradient_is_monotonic_increasing():
    df = generate_applicants(n=20_000, seed=42)
    by_moros = df[df["n_morosidad_reportes"] <= 3].groupby("n_morosidad_reportes")["default_12m"].mean()
    assert by_moros.is_monotonic_increasing


def test_is_deterministic_given_seed():
    df1 = generate_applicants(n=500, seed=7)
    df2 = generate_applicants(n=500, seed=7)
    assert df1["default_12m"].equals(df2["default_12m"])
