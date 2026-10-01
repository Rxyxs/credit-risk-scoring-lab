"""Nada en esta cadena mlops escribe un unico expediente por modelo -- cada
tecnica deja su propio archivo, con su propio esquema, en su propia
carpeta, y las conecta entre si solo por convencion de nombres o por
coincidencia de timestamp (12 dispara, 13 ingiere, 14 entrena, 15 decide,
16/18/20 despliegan, 22/23 reentrenan). Reconstruir "todo lo que paso con
este modelo" hoy significa abrir seis carpetas a mano. Esta tecnica hace
esa reconstruccion automaticamente, leyendo los mismos archivos e
inferiendo las mismas conexiones que haria una persona -- sin import de
codigo de ninguna tecnica anterior, solo sus archivos.

Dos tipos de union son honestos, y el expediente los distingue:
- **Exactas**: `promotion_decision.candidate_model == model_filename`,
  `model_lifecycle_events.new_champion == model_filename`. No hay
  ambiguedad posible, los nombres de archivo coinciden caracter por
  caracter.
- **Encadenadas, de a un salto por vez**: `champion_model.pkl` no nombra a
  ningun modelo con historia propia; el manifiesto de cutover (tecnica 20)
  apunta a `active_shadow_model.pkl` (el nombre fijo que usa el registro
  de la tecnica 16, nunca el original), y el nombre real con el que
  entreno y se promovio sobrevive un salto mas atras, en
  `registry_manifest.json` (`active_version`). Cada salto sigue siendo
  una union exacta por nombre de archivo -- solo hacen falta dos en vez
  de uno.
- **Inferidas por tiempo**: el disparador de drift (tecnica 12 o 22) que
  precede a un entrenamiento no tiene ninguna referencia directa al
  nombre del modelo que entrena despues -- se asume que el disparador
  mas reciente *antes* de `trained_at` es la causa. Es una inferencia
  razonable en un pipeline secuencial, pero sigue siendo una inferencia,
  y el expediente la marca como tal en vez de presentarla con la misma
  certeza que una union exacta.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

SHADOW_MODEL_RE = re.compile(r"^shadow_model_(?P<ts>.+)\.pkl$")
LIFECYCLE_TABLE = "model_lifecycle_events"

# (patron de archivo, campo de timestamp) para cada tipo de disparador de
# entrenamiento que este laboratorio produce en algun punto de la cadena.
TRIGGER_PATTERNS = (
    ("retrain_manifest_*.json", "generated_at"),      # tecnica 12
    ("retraining_trigger_*.json", "evaluated_at"),    # tecnica 22
)


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"


def _leer_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class ModelLineageTracker:
    def __init__(self):
        self.last_trace: dict | None = None

    # -- resolucion de identidad: champion_model.pkl no es un nombre de ----
    # archivo con historia propia, es una copia de algun shadow_model_*.pkl.
    def _resolver_origen(self, model_filename: str, reports_dirs: list[Path]):
        if SHADOW_MODEL_RE.match(model_filename):
            return model_filename, None

        for carpeta in reports_dirs:
            for manifest_path in sorted(Path(carpeta).glob("cutover_manifest_*.json")):
                manifest = _leer_json(manifest_path)
                if manifest is None or manifest.get("new_champion") != model_filename:
                    continue

                origen_path = Path(manifest.get("shadow_model_path", ""))
                if SHADOW_MODEL_RE.match(origen_path.name):
                    return origen_path.name, manifest

                # El caso real: 16-shadow-deployment copia el candidato
                # promovido a un nombre fijo (`active_shadow_model.pkl`),
                # no al nombre original con el que lo entreno la tecnica
                # 14 -- el cutover termina apuntando a ese nombre fijo, no
                # al identificador real del modelo. El nombre original
                # sobrevive en `registry_manifest.json`, al lado del
                # mismo archivo, bajo `active_version`.
                registry_manifest_path = origen_path.parent / "registry_manifest.json"
                registry_manifest = _leer_json(registry_manifest_path)
                if registry_manifest is not None:
                    version_activa = registry_manifest.get("active_version", "")
                    if SHADOW_MODEL_RE.match(version_activa):
                        return version_activa, manifest

                return origen_path.name or model_filename, manifest
        return model_filename, None

    def _find_training_metrics(self, shadow_filename: str, reports_dirs: list[Path]) -> dict | None:
        m = SHADOW_MODEL_RE.match(shadow_filename)
        if not m:
            return None
        nombre_metricas = f"shadow_metrics_{m.group('ts')}.json"
        for carpeta in reports_dirs:
            candidato = Path(carpeta) / nombre_metricas
            if candidato.exists():
                metricas = _leer_json(candidato)
                if metricas is not None:
                    return metricas
        return None

    def _find_promotion_decision(self, shadow_filename: str, reports_dirs: list[Path]) -> dict | None:
        for carpeta in reports_dirs:
            for decision_path in sorted(Path(carpeta).glob("promotion_decision_*.json")):
                decision = _leer_json(decision_path)
                if decision is not None and decision.get("candidate_model") == shadow_filename:
                    return decision
        return None

    def _find_drift_trigger(self, training_metrics: dict | None,
                            reports_dirs: list[Path]) -> dict | None:
        """El disparador mas reciente cuyo timestamp es anterior o igual a
        `training_metrics['trained_at']` -- una inferencia temporal, no una
        union exacta (ver el docstring del modulo)."""
        if training_metrics is None or not training_metrics.get("trained_at"):
            return None
        trained_at = training_metrics["trained_at"]

        candidatos = []
        for carpeta in reports_dirs:
            for patron, campo_tiempo in TRIGGER_PATTERNS:
                for path in Path(carpeta).glob(patron):
                    datos = _leer_json(path)
                    if datos is None:
                        continue
                    marca = datos.get(campo_tiempo)
                    if marca is not None and marca <= trained_at:
                        candidatos.append((marca, path.name, datos))

        if not candidatos:
            return None
        candidatos.sort(key=lambda t: t[0])
        _, nombre, datos = candidatos[-1]
        return {"source_file": nombre, "matched_by": "nearest_preceding_timestamp", **datos}

    def _find_data_origin(self, drift_trigger: dict | None) -> dict | None:
        if drift_trigger is None or "snapshot_path" not in drift_trigger:
            return None
        return {
            "snapshot_path": drift_trigger["snapshot_path"],
            "trigger_reason": drift_trigger.get("trigger_reason"),
            "features_affected": drift_trigger.get("features_affected"),
        }

    def _find_deployment_events(self, nombres_candidatos: set[str], db_path) -> list[dict]:
        """Filas de `model_lifecycle_events` donde el modelo aparece como
        `new_champion` o `previous_champion` -- unica union exacta posible
        contra la base, ordenada cronologicamente. Un `db_path` que todavia
        no existe no es un error: significa que este modelo nunca paso por
        un cutover, y el expediente lo refleja con una lista vacia."""
        db_path = Path(db_path)
        if not db_path.exists():
            return []

        try:
            con = duckdb.connect(str(db_path), read_only=True)
        except duckdb.Error as exc:
            logger.warning("No pude abrir '%s' de solo lectura: %s", db_path, exc)
            return []

        try:
            existe = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                [LIFECYCLE_TABLE],
            ).fetchone()[0] > 0
            if not existe:
                return []

            nombres = list(nombres_candidatos)
            placeholders = ", ".join(["?"] * len(nombres))
            filas = con.execute(
                f"SELECT * FROM {LIFECYCLE_TABLE} "
                f"WHERE new_champion IN ({placeholders}) OR previous_champion IN ({placeholders}) "
                f"ORDER BY promoted_at",
                nombres + nombres,
            ).fetchdf()
        finally:
            con.close()

        return filas.to_dict(orient="records")

    def trace_model_lineage(self, model_filename: str, db_path, reports_dirs) -> dict:
        reports_dirs = [Path(d) for d in reports_dirs]

        origen, cutover_manifest = self._resolver_origen(model_filename, reports_dirs)
        training_metrics = self._find_training_metrics(origen, reports_dirs)
        promotion_decision = self._find_promotion_decision(origen, reports_dirs)
        drift_trigger = self._find_drift_trigger(training_metrics, reports_dirs)
        data_origin = self._find_data_origin(drift_trigger)

        nombres_candidatos = {model_filename, origen}
        deployment_events = self._find_deployment_events(nombres_candidatos, db_path)
        if cutover_manifest is not None:
            # El manifiesto del propio cutover ya es un registro exacto de
            # un evento de despliegue -- se agrega aunque la fila de
            # DuckDB correspondiente tambien haya aparecido, para no
            # perder el detalle completo (la fila de la tabla solo tiene
            # 5 columnas; el manifiesto tiene el resto).
            deployment_events = [{"source": "cutover_manifest", **cutover_manifest}] + deployment_events

        trazado = any([training_metrics, promotion_decision, deployment_events, drift_trigger])

        resultado = {
            "model_id": model_filename,
            "resolved_origin": origen if origen != model_filename else None,
            "status": "TRACED" if trazado else "UNTRACED",
            "data_origin": data_origin,
            "drift_trigger": drift_trigger,
            "training_metrics": training_metrics,
            "promotion_decision": promotion_decision,
            "deployment_events": deployment_events,
            "traced_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        self.last_trace = resultado
        return dict(resultado)

    def generate_governance_manifest(self, model_filename: str, output_dir) -> Path:
        if self.last_trace is None or self.last_trace.get("model_id") != model_filename:
            raise RuntimeError(
                f"no hay un trazado vigente para '{model_filename}' en esta instancia -- "
                "llama trace_model_lineage() con el mismo model_filename primero.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nombre_seguro = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(model_filename).stem)
        manifest_path = output_dir / f"model_lineage_{nombre_seguro}_{_timestamp()}.json"
        manifest_path.write_text(json.dumps(self.last_trace, indent=2, default=str))
        return manifest_path
