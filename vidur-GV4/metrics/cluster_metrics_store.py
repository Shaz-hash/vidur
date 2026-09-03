import copy
import json
import os
from typing import Dict, List, Optional

import pandas as pd
import wandb

from vidur.config import SimulationConfig
from vidur.entities.batch import Batch
from vidur.entities.replica import Replica
from vidur.entities.request import Request
from vidur.metrics.data_structures.cdf_sketch import CDFSketch
from vidur.metrics.data_structures.data_series import DataSeries
from vidur.metrics.metrics import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    BatchMetricsTimeSeries,
    CpuOperationMetrics,
    OperationMetrics,
    RequestMetricsTimeDistributions,
    RequestMetricsTimeSeries,
    TokenMetricsTimeDistribution,
    TokenMetricsTimeSeries,
)
from vidur.metrics.replica_metrics_store import ReplicaMetricsStore
from vidur.types.replica_id import ReplicaId


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
TIME_STR_MS = "Time (ms)"
TIME_STR = "Time (sec)"
COUNT_STR = "Count"
MEMORY_USAGE_STR = "Memory Usage (%)"


class ClusterMetricsStore:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        replicas: Dict[ReplicaId, Replica],
    ):
        self._simulation_config = simulation_config
        self._config = self._simulation_config.metrics_config
        self._replicas = replicas

        """
        TODO: We need to use a minimal `ClusterMetricsStore`.
        Explicitly not collect batch, utilization, operation metrics at cluster level as they are replica specific.
        """
        simulation_config_copy = copy.deepcopy(simulation_config)
        simulation_config_copy.metrics_config.store_batch_metrics = False
        simulation_config_copy.metrics_config.store_utilization_metrics = False
        simulation_config_copy.metrics_config.store_operation_metrics = False
        self._cluster_metric_store = ReplicaMetricsStore(
            simulation_config=simulation_config_copy,
        )

        # We use str(replica_id) as the key to avoid JSON encoding issues inside wandb.log
        self._replica_metric_stores: Dict[str, ReplicaMetricsStore] = {}
        self._replica_metric_store_aliases: Dict[str, ReplicaMetricsStore] = {}

        for replica_id in replicas.keys():
            store = ReplicaMetricsStore(
                simulation_config=self._simulation_config,
                replica_id=replica_id,
            )
            canonical_key = self._canonical_replica_key(replica_id)
            self._replica_metric_stores[canonical_key] = store
            self._register_metric_aliases_for_replica(replica_id, store)

        self._wandb_project = self._config.wandb_project
        self._wandb_group = self._config.wandb_group
        self._wandb_run_name = self._config.wandb_run_name

        self._init_wandb()

    def register_replica_alias(
        self,
        alias_key: object,
        replica_id: Optional[ReplicaId] = None,
        store: Optional[ReplicaMetricsStore] = None,
    ) -> None:
        if store is None and replica_id is not None:
            try:
                store = self._get_replica_metrics_store(replica_id)
            except KeyError:
                store = None

        if store is None:
            stores = list(self._replica_metric_stores.values())
            if stores:
                first_store = stores[0]
                if all(s is first_store for s in stores):
                    store = first_store

        if store is None:
            alias_stores = list(self._replica_metric_store_aliases.values())
            if alias_stores:
                first_store = alias_stores[0]
                if all(s is first_store for s in alias_stores):
                    store = first_store

        if store is None and self._replica_metric_stores:
            store = next(iter(self._replica_metric_stores.values()))

        if store is not None:
            self._register_metric_alias(alias_key, store)

    def _canonical_replica_key(self, replica_id: ReplicaId) -> str:
        if hasattr(replica_id, "id"):
            try:
                return str(int(replica_id.id))
            except Exception:
                pass
        return str(replica_id)

    def _register_metric_aliases_for_replica(
        self, replica_id: ReplicaId, store: ReplicaMetricsStore
    ) -> None:
        keys = {
            self._canonical_replica_key(replica_id),
            str(replica_id),
            repr(replica_id),
        }
        if hasattr(replica_id, "id"):
            try:
                keys.add(str(int(replica_id.id)))
            except Exception:
                pass
        if hasattr(replica_id, "_id"):
            try:
                keys.add(str(int(replica_id._id)))
            except Exception:
                pass
        for key in keys:
            self._register_metric_alias(key, store)

    def _register_metric_alias(
        self, alias_key: object, store: ReplicaMetricsStore
    ) -> None:
        self._replica_metric_store_aliases[str(alias_key)] = store

    def _get_replica_metrics_store(self, replica_id: ReplicaId) -> ReplicaMetricsStore:
        canonical = self._canonical_replica_key(replica_id)
        if canonical in self._replica_metric_stores:
            return self._replica_metric_stores[canonical]

        candidate_keys = [
            canonical,
            str(replica_id),
            repr(replica_id),
        ]
        if hasattr(replica_id, "id"):
            try:
                candidate_keys.append(str(int(replica_id.id)))
            except Exception:
                pass
        if hasattr(replica_id, "_id"):
            try:
                candidate_keys.append(str(int(replica_id._id)))
            except Exception:
                pass

        for key in candidate_keys:
            store = self._replica_metric_store_aliases.get(str(key))
            if store:
                return store

        stores = list(self._replica_metric_stores.values())
        if stores:
            first_store = stores[0]
            if all(s is first_store for s in stores):
                return first_store

        alias_stores = list(self._replica_metric_store_aliases.values())
        if alias_stores:
            first_store = alias_stores[0]
            if all(s is first_store for s in alias_stores):
                return first_store

        known_keys = sorted(self._replica_metric_store_aliases.keys())
        raise KeyError(
            f"Replica metrics store not found for {replica_id}. "
            f"Known keys include: {known_keys[:10]}"
        )

    def _init_wandb(self):
        if (
            not self._config.write_metrics
            or not self._wandb_project
            or not self._wandb_group
        ):
            return

        wandb.init(
            project=self._wandb_project,
            group=self._wandb_group,
            name=self._wandb_run_name,
            config=self._simulation_config.to_dict(),
        )


    def _store_batch_request_details(self, base_plot_path: str):
        if not self._config.store_batch_metrics:
            return

        # Keep behavior consistent with batch_metrics.csv:
        if self._config.keep_individual_batch_metrics:
            brd_df = pd.DataFrame()
            for _, store in self._replica_metric_stores.items():
                brd_df = pd.concat(
                    [brd_df, store.get_batch_request_details_df()],
                    ignore_index=True
                )

            # Only write if we have rows
            if not brd_df.empty:
                self._save_as_csv(
                    df=brd_df,
                    base_path=self._config.output_dir,
                    file_name="Batch_requests_details",  # -> Batch_requests_details.csv
                )



    def _save_as_csv(
        self,
        df: pd.DataFrame,
        base_path: str,
        file_name: str,
    ):
        os.makedirs(base_path, exist_ok=True)
        # Print only upto 6 decimal places (micros precision) to reduce csv size
        df.to_csv(f"{base_path}/{file_name}.csv", float_format="%.6f", index=False)
        if wandb.run and self._config.save_table_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{file_name}_table": wand_table}, step=0)

    def _save_as_json(self, data, base_path: str, file_name: str):
        os.makedirs(base_path, exist_ok=True)
        with open(f"{base_path}/{file_name}.json", "w") as f:
            json.dump(data, f)

        if wandb.run and self._config.save_table_to_wandb:
            wandb.log({f"{file_name}": data}, step=0)

    def _store_request_metrics(self, base_plot_path: str):
        if not self._config.store_request_metrics:
            return

        request_metrics_df = pd.DataFrame()
        for replica_id, store in self._replica_metric_stores.items():
            request_metrics_df = pd.concat(
                [request_metrics_df, store.get_request_metrics_df()]
            )
        request_metrics_df.sort_values(by=REQUEST_ID_STR, inplace=True)
        self._save_as_csv(
            df=request_metrics_df,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        # Log prefix cache metrics
        prefix_cache_metrics = {}
        for replica_id, store in self._replica_metric_stores.items():
            prefix_cache_stats = store.get_prefix_cache_stats()
            prefix_cache_metrics[replica_id] = prefix_cache_stats
        self._save_as_json(
            data=prefix_cache_metrics,
            base_path=base_plot_path,
            file_name="replica_prefix_cache_metrics",
        )

        # Print replica wise metrics
        if len(self._replica_metric_stores) > 1:
            for metric_name in RequestMetricsTimeDistributions:
                replica_wise_dict = {}
                for replica_id, store in self._replica_metric_stores.items():
                    replica_wise_dict[replica_id] = (
                        store._request_metrics_time_distributions[metric_name]
                    )
                DataSeries.plot_cdfs(
                    replica_wise_dict,
                    base_plot_path,
                    f"{metric_name.value}_replicawise",
                    y_axis_label=TIME_STR,
                    save_plot=self._config.store_plots,
                )

            for metric_name in RequestMetricsTimeSeries:
                replica_wise_dict = {}
                for replica_id, store in self._replica_metric_stores.items():
                    replica_wise_dict[replica_id] = store._request_metrics_time_series[
                        metric_name
                    ]
                DataSeries.plot_steps(
                    replica_wise_dict,
                    base_plot_path,
                    f"{metric_name.value}_timeseries_replicawise",
                    y_axis_label=TIME_STR,
                    save_plot=self._config.store_plots,
                )

    def _store_batch_metrics(self, base_plot_path: str):
        if not self._config.store_batch_metrics:
            return

        if self._config.keep_individual_batch_metrics:
            batch_metrics_df = pd.DataFrame()
            for replica_id, store in self._replica_metric_stores.items():
                batch_metrics_df = pd.concat(
                    [batch_metrics_df, store.get_batch_metrics_df()]
                )
            self._save_as_csv(
                df=batch_metrics_df,
                base_path=self._config.output_dir,
                file_name="batch_metrics",
            )

        for metric_name in BatchMetricsTimeDistribution:
            y_axis_label = (
                TIME_STR_MS if "model_execution" in metric_name.value else TIME_STR
            )

            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._batch_metrics_time_distribution[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                metric_name.value,
                y_axis_label,
                save_plot=self._config.store_plots,
            )

        for metric_name in BatchMetricsCountDistribution:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._batch_metrics_count_distribution[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                metric_name.value,
                y_axis_label=COUNT_STR,
                save_plot=self._config.store_plots,
            )

        # if self._config.keep_individual_batch_metrics:
        #     for metric_name in BatchMetricsTimeSeries:
        #         replica_wise_dict = {}
        #         for replica_id, store in self._replica_metric_stores.items():
        #             replica_wise_dict[replica_id] = store._batch_metrics_time_series[
        #                 metric_name
        #             ]
        #         DataSeries.plot_steps(
        #             replica_wise_dict,
        #             base_plot_path,
        #             f"{metric_name.value}_replicawise",
        #             y_axis_label=metric_name.value,
        #             save_plot=self._config.store_plots,
        #             y_cumsum=False,
        #         )

    def _store_token_metrics(self, base_plot_path: str):
        if not self._config.store_token_completion_metrics:
            return
        for metric_name in TokenMetricsTimeDistribution:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._token_metrics_time_distribution[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                metric_name.value,
                y_axis_label=TIME_STR,
                save_plot=self._config.store_plots,
            )

        for metric_name in TokenMetricsTimeSeries:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._token_metrics_time_series[
                    metric_name
                ]
            DataSeries.plot_steps(
                replica_wise_dict,
                base_plot_path,
                f"{metric_name.value}_timeseries_replicawise",
                y_axis_label=COUNT_STR,
                save_plot=self._config.store_plots,
            )

    def _store_operation_metrics(self, base_plot_path: str):
        if not self._config.store_operation_metrics:
            return

        if self._config.keep_individual_batch_metrics:
            op_metrics_df = pd.DataFrame()
            for replica_id, store in self._replica_metric_stores.items():
                op_metrics_df = pd.concat(
                    [op_metrics_df, store.get_operation_metrics_df()]
                )
            self._save_as_csv(
                df=op_metrics_df,
                base_path=self._config.output_dir,
                file_name="operation_metrics",
            )

        for metric_name in OperationMetrics:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._operation_metrics[metric_name]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                f"{metric_name.value}_execution_time",
                y_axis_label=TIME_STR_MS,
                save_plot=self._config.store_plots,
            )

        for metric_name in CpuOperationMetrics:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._cpu_operation_metrics[
                    metric_name
                ]
            CDFSketch.plot_cdfs(
                replica_wise_dict,
                base_plot_path,
                f"{metric_name.value}_execution_time",
                y_axis_label=TIME_STR_MS,
                save_plot=self._config.store_plots,
            )

    def _store_utilization_metrics(self, base_plot_path: str):
        if not self._config.store_utilization_metrics:
            return

        replica_memory_usage = {}
        replica_busy_time = {}
        replica_mfu = {}
        for replica_id, store in self._replica_metric_stores.items():
            replica_memory_usage[str(replica_id)] = (
                store._replica_memory_usage.get_stats("replica_memory_usage")
            )
            replica_busy_time[str(replica_id)] = store.get_replica_busy_time()
            replica_mfu[str(replica_id)] = store.get_replica_mfu()

        self._save_as_json(replica_memory_usage, base_plot_path, "replica_memory_usage")
        self._save_as_json(replica_busy_time, base_plot_path, "replica_busy_time")
        self._save_as_json(replica_mfu, base_plot_path, "replica_mfu")

        if self._config.keep_individual_batch_metrics:
            replica_wise_dict = {}
            for replica_id, store in self._replica_metric_stores.items():
                replica_wise_dict[replica_id] = store._replica_memory_usage_per_batch
            # TODO: Fix perf and enable plotting the memory usage wrt time
            DataSeries.plot_steps(
                replica_wise_dict,
                base_plot_path,
                "replica_memory_usage_time_series",
                y_axis_label=MEMORY_USAGE_STR,
                save_plot=False,
                y_cumsum=False,
            )

    @if_write_metrics
    def plot(self, sim_time: float) -> None:
        dir_plot_path = f"{self._config.output_dir}/plots"
        os.makedirs(dir_plot_path, exist_ok=True)

        self._cluster_metric_store.store_metrics(dir_plot_path, sim_time)
        self._store_request_metrics(dir_plot_path)
        self._store_batch_metrics(dir_plot_path)

        # NEW: write per-(batch, request) details CSV in the same pass as batch metrics
        self._store_batch_request_details(dir_plot_path)

        self._store_token_metrics(dir_plot_path)
        self._store_operation_metrics(dir_plot_path)
        self._store_utilization_metrics(dir_plot_path)

    def on_batch_end(
        self, time: float, batch, replica_id: ReplicaId, memory_usage_percent: float
    ):
        self._cluster_metric_store.on_batch_end(time, batch, memory_usage_percent)
        self._get_replica_metrics_store(replica_id).on_batch_end(
            time, batch, memory_usage_percent
        )

    def on_batch_stage_end(
        self, batch_stage, time: float, replica_id: ReplicaId, stage_id: int
    ):
        self._get_replica_metrics_store(replica_id).on_batch_stage_end(
            batch_stage, time, stage_id
        )

    def on_replica_schedule(
        self,
        time: float,
        replica_id: ReplicaId,
        batches: List[Batch],
        memory_usage_percent: float,
    ):
        self._get_replica_metrics_store(replica_id).on_replica_schedule(
            time, memory_usage_percent
        )
        newly_scheduled_requests = [
            request
            for batch in batches
            for request in batch.requests
            if request.scheduled_at == time
        ]
        for request in newly_scheduled_requests:
            self._get_replica_metrics_store(request.replica_id).on_request_arrival(
                request
            )

    def on_replica_stage_schedule(
        self,
        time: float,
        replica_id: ReplicaId,
        stage_id: int,
        batch_stage,
        execution_time: float,
    ):
        self._get_replica_metrics_store(replica_id).on_replica_stage_schedule(
            time, stage_id, batch_stage, execution_time
        )

    def on_request_arrival(self, request: Request):
        self._cluster_metric_store.on_request_arrival(request)

    def on_request_end(self, request: Request):
        self._cluster_metric_store.on_request_end(request)
        self._get_replica_metrics_store(request.replica_id).on_request_end(request)
