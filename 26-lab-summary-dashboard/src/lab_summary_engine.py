"""El ultimo eslabon: no agrega ningun comportamiento nuevo al laboratorio,
lo audita. Lee las mismas bases DuckDB y los mismos archivos que las
tecnicas 12 a 25 ya dejaron atras -- sin import de su codigo, el mismo
criterio que toda esta cadena mantiene desde `19-canary-monitoring` -- y
arma un unico informe que responde la pregunta que ningun archivo suelto
responde por si solo: ¿esta sano el laboratorio, en este momento, en
conjunto?

No existe una unica base "central": `13-feature-store-duckdb`,
`16-shadow-deployment`, `18-canary-deployment`, `20-full-promotion-cutover`
y `25-api-inference-service` cada una tiene su propio archivo DuckDB con
su propia tabla. `audit_database_integrity` por eso es generico -- recibe
*una* ruta e introspecciona lo que esa base realmente tiene
(`information_schema.tables`), en vez de asumir una lista fija de nombres
de tabla -- y `generate_master_summary` es quien lo llama una vez por cada
base real conocida y combina los resultados.
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import duckdb

# Columnas candidatas de timestamp, en el orden en que aparecen en las
# tablas reales de este laboratorio (ver 16/18/20/23/25) -- la primera que
# la tabla realmente tenga gana.
TIMESTAMP_COLUMN_CANDIDATES = (
    "promoted_at", "timestamp", "logged_at", "generated_at",
    "evaluated_at", "activated_at", "trained_at", "updated_at",
)

# Rutas reales de este laboratorio, no inventadas: cada una es donde la
# tecnica correspondiente realmente escribe, confirmado corriendo esa
# tecnica de punta a punta durante la construccion de esta.
DEFAULT_FEATURE_STORE_DB = "../13-feature-store-duckdb/outputs/offline_store.duckdb"
DEFAULT_DUAL_INFERENCE_DB = "../16-shadow-deployment/outputs/dual_inference_logs.duckdb"
DEFAULT_CANARY_DB = "../18-canary-deployment/outputs/canary_predictions.duckdb"
DEFAULT_API_LOG_DB = "../25-api-inference-service/outputs/api_inference_log.duckdb"

DEFAULT_CHAMPION_DIR = "../20-full-promotion-cutover/outputs/models/champion"
DEFAULT_SHADOW_DIRS = (
    "../14-shadow-model-training/outputs/models",
    "../23-automated-retraining-pipeline/outputs/models",
)
DEFAULT_LINEAGE_DIR = "../24-model-lineage-governance/outputs/manifests"
DEFAULT_TELEMETRY_DIR = "../21-post-cutover-telemetry/outputs/reports"
DEFAULT_TRIGGER_DIR = "../22-automated-retraining-trigger/outputs/manifests"


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"


def _leer_json_seguro(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _es_pickle_valido(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        return False


class LabSummaryEngine:
    def __init__(self):
        self.last_summary: dict | None = None

    # -- bases DuckDB ------------------------------------------------------
    def audit_database_integrity(self, db_path) -> dict:
        """Lo que esa base realmente tiene, tabla por tabla: conteo de
        filas y la fecha del evento mas reciente (si la tabla tiene alguna
        columna de timestamp reconocible). Un archivo que no existe no es
        un error -- es una tecnica que todavia no corrio -- asi que nunca
        se intenta `duckdb.connect` contra un archivo que no esta: un
        archivo faltante no tiene directorio padre que crear, tiene un
        archivo entero que falta, y `mkdir` no soluciona eso."""
        db_path = Path(db_path)
        if not db_path.exists():
            return {"db_path": str(db_path), "exists": False, "tables": {}}

        try:
            con = duckdb.connect(str(db_path), read_only=True)
        except duckdb.Error as exc:
            return {"db_path": str(db_path), "exists": True, "error": str(exc), "tables": {}}

        try:
            nombres = [fila[0] for fila in con.execute(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()]

            tablas = {}
            for nombre in nombres:
                conteo = con.execute(f"SELECT count(*) FROM {nombre}").fetchone()[0]
                tablas[nombre] = {
                    "row_count": int(conteo),
                    "latest_event_at": self._ultima_fecha(con, nombre),
                }
        finally:
            con.close()

        return {"db_path": str(db_path), "exists": True, "tables": tablas}

    def _ultima_fecha(self, con, tabla: str) -> str | None:
        columnas = {
            fila[0] for fila in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [tabla],
            ).fetchall()
        }
        for candidata in TIMESTAMP_COLUMN_CANDIDATES:
            if candidata in columnas:
                resultado = con.execute(f"SELECT max({candidata}) FROM {tabla}").fetchone()[0]
                if resultado is not None:
                    return str(resultado)
        return None

    # -- registro de artefactos ---------------------------------------------
    def audit_artifact_registry(self, registry_dir) -> dict:
        """`champion_model.pkl` (si esta en esta carpeta), cualquier
        `shadow_model_*.pkl`, y cualquier `model_lineage_*.json` -- cada
        uno marcado como valido o no (un .pkl que no deserializa, un .json
        que no parsea), nunca asumido valido solo porque existe."""
        directorio = Path(registry_dir)
        if not directorio.is_dir():
            return {
                "registry_dir": str(directorio), "exists": False,
                "champion": None, "shadow_candidates": [], "lineage_reports": [],
            }

        champion = None
        champion_path = directorio / "champion_model.pkl"
        if champion_path.exists():
            champion = {
                "path": str(champion_path),
                "valid_pickle": _es_pickle_valido(champion_path),
                "size_bytes": champion_path.stat().st_size,
            }

        shadow_candidates = [
            {"path": str(p), "valid_pickle": _es_pickle_valido(p), "size_bytes": p.stat().st_size}
            for p in sorted(directorio.glob("shadow_model_*.pkl"))
        ]

        lineage_reports = []
        for p in sorted(directorio.glob("model_lineage_*.json")):
            datos = _leer_json_seguro(p)
            lineage_reports.append({
                "path": str(p),
                "valid_json": datos is not None,
                "status": datos.get("status") if datos else None,
            })

        return {
            "registry_dir": str(directorio), "exists": True,
            "champion": champion, "shadow_candidates": shadow_candidates,
            "lineage_reports": lineage_reports,
        }

    # -- lo que no vive en una tabla: telemetria y disparadores -------------
    def _ultimo_reporte_telemetria(self, telemetry_dir) -> dict | None:
        directorio = Path(telemetry_dir)
        if not directorio.is_dir():
            return None
        candidatos = sorted(directorio.glob("post_cutover_telemetry_*.json"))
        if not candidatos:
            return None
        datos = _leer_json_seguro(candidatos[-1])
        if datos is None:
            return None
        rendimiento = datos.get("realized_performance") or {}
        drift = datos.get("prediction_drift") or {}
        return {
            "source_file": candidatos[-1].name,
            "status": rendimiento.get("status"),
            "realized_roc_auc": rendimiento.get("realized_roc_auc"),
            "psi": drift.get("psi"),
        }

    def _ultimo_estado_disparador(self, trigger_dir) -> dict | None:
        directorio = Path(trigger_dir)
        if not directorio.is_dir():
            return None
        candidatos = sorted(directorio.glob("retraining_trigger_*.json"))
        if not candidatos:
            return None
        datos = _leer_json_seguro(candidatos[-1])
        if datos is None:
            return None
        return {
            "source_file": candidatos[-1].name,
            "trigger_activated": datos.get("trigger_activated"),
            "reasons": datos.get("reasons", []),
        }

    # -- el informe maestro --------------------------------------------------
    def _evaluar_estado_global(self, champion: dict | None, estado_disparador: dict | None) -> str:
        """DEGRADED, explicito y por una razon concreta -- no una mezcla
        de señales promediadas en un numero: o no hay Champion valido
        sirviendo, o hay un disparador de reentrenamiento activo que
        nadie atendio todavia. Cualquier otra combinacion es HEALTHY."""
        if champion is None or not champion.get("valid_pickle"):
            return "DEGRADED"
        if estado_disparador is not None and estado_disparador.get("trigger_activated") is True:
            return "DEGRADED"
        return "HEALTHY"

    def generate_master_summary(self, db_path, outputs_dir) -> dict:
        bases = {
            "lifecycle_events": self.audit_database_integrity(db_path),
            "feature_store": self.audit_database_integrity(DEFAULT_FEATURE_STORE_DB),
            "dual_inference": self.audit_database_integrity(DEFAULT_DUAL_INFERENCE_DB),
            "canary_routing": self.audit_database_integrity(DEFAULT_CANARY_DB),
            "api_inference": self.audit_database_integrity(DEFAULT_API_LOG_DB),
        }

        registro_champion = self.audit_artifact_registry(DEFAULT_CHAMPION_DIR)
        registros_shadow = [self.audit_artifact_registry(d) for d in DEFAULT_SHADOW_DIRS]
        registro_linaje = self.audit_artifact_registry(DEFAULT_LINEAGE_DIR)

        total_ejecuciones = sum(
            tabla["row_count"]
            for base in bases.values()
            for tabla in base["tables"].values()
        )

        ultima_telemetria = self._ultimo_reporte_telemetria(DEFAULT_TELEMETRY_DIR)
        estado_disparador = self._ultimo_estado_disparador(DEFAULT_TRIGGER_DIR)
        champion_en_servicio = registro_champion.get("champion")

        resumen = {
            "lab_status": self._evaluar_estado_global(champion_en_servicio, estado_disparador),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "total_executions": total_ejecuciones,
            "databases": bases,
            "champion_in_service": champion_en_servicio,
            "shadow_candidate_registries": registros_shadow,
            "lineage_registry": registro_linaje,
            "latest_telemetry": ultima_telemetria,
            "retraining_trigger_status": estado_disparador,
        }
        self.last_summary = resumen

        outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        reporte_path = outputs_dir / f"lab_summary_report_{_timestamp()}.json"
        reporte_path.write_text(json.dumps(resumen, indent=2, default=str))

        markdown_path = outputs_dir / "LAB_SUMMARY.md"
        markdown_path.write_text(self._render_markdown(resumen))

        return {
            "summary": resumen,
            "report_path": str(reporte_path),
            "markdown_path": str(markdown_path),
        }

    def _render_markdown(self, resumen: dict) -> str:
        lineas = [
            "# Lab Summary",
            "",
            f"**Estado global:** `{resumen['lab_status']}`  ",
            f"**Generado:** {resumen['generated_at']}  ",
            f"**Ejecuciones totales registradas:** {resumen['total_executions']}",
            "",
            "## Champion en servicio",
            "",
        ]
        champion = resumen["champion_in_service"]
        if champion:
            lineas.append(f"- Ruta: `{champion['path']}`")
            lineas.append(f"- Pickle valido: {champion['valid_pickle']}")
            lineas.append(f"- Tamano: {champion['size_bytes']} bytes")
        else:
            lineas.append("- **Sin Champion detectado en el registro.**")

        lineas += ["", "## Bases de datos auditadas", ""]
        lineas.append("| Base | Existe | Tablas | Filas totales |")
        lineas.append("|---|---|---|---|")
        for nombre, base in resumen["databases"].items():
            filas_totales = sum(t["row_count"] for t in base["tables"].values())
            lineas.append(f"| {nombre} | {base['exists']} | {len(base['tables'])} | {filas_totales} |")

        lineas += ["", "## Telemetria mas reciente", ""]
        telemetria = resumen["latest_telemetry"]
        if telemetria:
            lineas.append(f"- Archivo: `{telemetria['source_file']}`")
            lineas.append(f"- Estado: {telemetria['status']}")
            lineas.append(f"- ROC-AUC realizado: {telemetria['realized_roc_auc']}")
            lineas.append(f"- PSI: {telemetria['psi']}")
        else:
            lineas.append("- Sin reportes de telemetria todavia.")

        lineas += ["", "## Disparador de reentrenamiento", ""]
        disparador = resumen["retraining_trigger_status"]
        if disparador:
            lineas.append(f"- Archivo: `{disparador['source_file']}`")
            lineas.append(f"- Activo: {disparador['trigger_activated']}")
            if disparador["reasons"]:
                for razon in disparador["reasons"]:
                    lineas.append(f"  - {razon}")
        else:
            lineas.append("- Sin disparadores todavia.")

        lineas.append("")
        return "\n".join(lineas)
