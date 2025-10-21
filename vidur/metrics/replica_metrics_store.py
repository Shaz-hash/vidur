from typing import Dict, List, Optional

import pandas as pd

from vidur.config import SimulationConfig
from vidur.entities import Batch, BatchStage, ExecutionTime, Request
from vidur.logger import init_logger
from vidur.metrics.data_structures.cdf_sketch import CDFSketch
from vidur.metrics.data_structures.data_series import DataSeries
from vidur.metrics.data_structures.series_average_meter import SeriesAverageMeter
from vidur.metrics.metrics import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    BatchMetricsTimeSeries,
    CpuOperationMetrics,
    OperationMetrics,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    RequestMetricsTimeSeries,
    TokenMetricsTimeDistribution,
    TokenMetricsTimeSeries,
)
from vidur.utils.mfu_calculator import MFUCalculator

logger = init_logger(__name__)


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
BATCH_ID_STR = "Batch Id"
MEMORY_USAGE_STR = "Memory Usage (%)"
BUSY_TIME_PERCENT = "Busy Time (%)"
UTILIZATION_STR = "Utilization (%)"
OPERATION_STR = "Operation"
TIME_STR_MS = "Time (ms)"


class ReplicaMetricsStore:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        replica_id: Optional[int] = None,
    ) -> None:
        self._replica_id = replica_id
        self._simulation_config = simulation_config

        # copy config
        self._config = self._simulation_config.metrics_config
        self._num_replicas = self._simulation_config.cluster_config.num_replicas
        self._num_pipeline_stages = (
            self._simulation_config.cluster_config.replica_config.num_pipeline_stages
        )
        self._last_request_arrived_at = 0.0

        # Initialise request metrics
        self._request_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._request_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
        self._request_metrics_time_series: Dict[
            RequestMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeSeries:
            self._request_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._request_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._request_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise batch metrics

        # --> new Collect per-(batch, request) rows for Batch_requests_details.csv
        self._batch_request_details_rows: List[dict] = []
      

        # Track which requests have ever been scheduled (if scheduler doesn’t pass it)
        self._ever_scheduled_req_ids: set[str] = set()

        self._batch_metrics_count_distribution: Dict[
            BatchMetricsCountDistribution, CDFSketch
        ] = {}
        self._batch_metrics_count_distribution_per_batch: Dict[
            BatchMetricsCountDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._batch_metrics_count_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, CDFSketch
        ] = {}
        self._batch_metrics_time_distribution_per_batch: Dict[
            BatchMetricsTimeDistribution, DataSeries
        ] = {}

  
        # --- NEW / EXTEND: extra per-batch columns to be merged into batch_metrics.csv
        self._batch_extras_per_batch: Dict[str, DataSeries] = {
            "kv_free_blocks_before": DataSeries(
                BATCH_ID_STR, "kv_free_blocks_before",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
            "kv_free_blocks_after": DataSeries(
                BATCH_ID_STR, "kv_free_blocks_after",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
            "batch_request_ids": DataSeries(
                BATCH_ID_STR, "batch_request_ids",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),

            # --- NEW snapshot counters at batch start ---
            "num_requests_not_selected": DataSeries(
                BATCH_ID_STR, "num_requests_not_selected",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
            "num_requests_initiated_not_completed": DataSeries(
                BATCH_ID_STR, "num_requests_initiated_not_completed",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
            "num_decode_phase_total": DataSeries(
                BATCH_ID_STR, "num_decode_phase_total",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
            "num_prefill_queue_total": DataSeries(
                BATCH_ID_STR, "num_prefill_queue_total",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
            "num_not_initiated_in_queue": DataSeries(
                BATCH_ID_STR, "num_not_initiated_in_queue",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
            "total_requests_batch_start": DataSeries(
                BATCH_ID_STR, "total_requests_batch_start",
                self._config.subsamples, self._config.save_table_to_wandb, self._config.store_plots
            ),
        }



        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._batch_metrics_time_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
        self._batch_metrics_time_series: Dict[BatchMetricsTimeSeries, DataSeries] = {}
        for metric_name in BatchMetricsTimeSeries:
            self._batch_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise token metrics
        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
        self._token_metrics_time_series: Dict[TokenMetricsTimeSeries, DataSeries] = {}
        for metric_name in TokenMetricsTimeSeries:
            self._token_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise operation metrics
        self._operation_metrics: Dict[OperationMetrics, CDFSketch] = {}
        self._operation_metrics_per_batch: Dict[OperationMetrics, DataSeries] = {}
        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._cpu_operation_metrics: Dict[CpuOperationMetrics, CDFSketch] = {}
        self._cpu_operation_metrics_per_batch: Dict[CpuOperationMetrics, DataSeries] = (
            {}
        )
        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._cpu_operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise utilization metrics
        self._replica_memory_usage = SeriesAverageMeter(
            TIME_STR,
            MEMORY_USAGE_STR,
            self._config.save_table_to_wandb,
        )
        self._replica_memory_usage.put(0, 0)
        self._replica_memory_usage_per_batch = DataSeries(
            TIME_STR,
            MEMORY_USAGE_STR,
            self._config.subsamples,
            self._config.save_table_to_wandb,
            self._config.store_plots,
        )
        self._replica_busy_time = []
        self._replica_mfu = []
        self._mfu_calculator = MFUCalculator(
            self._simulation_config.cluster_config.replica_config
        )
        for stage_idx in range(self._num_pipeline_stages):
            self._replica_busy_time.append(
                SeriesAverageMeter(
                    TIME_STR,
                    BUSY_TIME_PERCENT,
                    save_table_to_wandb=self._config.save_table_to_wandb,
                )
            )
            self._replica_busy_time[stage_idx].put(0, 0)

            self._replica_mfu.append(
                SeriesAverageMeter(
                    TIME_STR,
                    UTILIZATION_STR,
                    save_table_to_wandb=self._config.save_table_to_wandb,
                )
            )
            self._replica_mfu[stage_idx].put(0, 0)


    def add_batch_request_details(self, batch: Batch):
        """
        Build per-request rows AFTER this batch executes.

        - If the scheduler attached batch._request_details_rows (full snapshot across live requests),
        prefer those and just tag the replica.
        - Otherwise, fall back to synthesizing rows only for requests in this batch.
        """¬
        # 1) Prefer scheduler-provided comprehensive rows
        if hasattr(batch, "_request_details_rows") and batch._request_details_rows:
            for row in batch._request_details_rows:
                row["replica"] = self._replica_id
                # maintain ever-scheduled set if the scheduler sent rows for requests in the batch
                if "request_id" in row:
                    self._ever_scheduled_req_ids.add(row["request_id"])
            self._batch_request_details_rows.extend(batch._request_details_rows)
            return

        # 2) Fallback: synthesize minimal info for requests in *this* batch only
        for req in getattr(batch, "requests", []):
            was_ever_scheduled = (req.id in self._ever_scheduled_req_ids)

            row = {
                "batch_id": batch.id,
                "request_id": req.id,
                "slo_type": "TTFT" if not req.is_prefill_complete else "TBT",
                "slo_remaining_ms": 0,
                "only_in_queue": (not was_ever_scheduled),
                "added_in_last_batch": True,  # fallback path cannot know others
                "request_type_after_batch": "decode" if getattr(req, "has_started_decode", False) else "prefill",
                "kv_context_len_tokens": getattr(req, "num_processed_tokens", 0),
                "prefill_len_tokens": getattr(req, "num_prefill_tokens", 0),
                "prefill_remaining_tokens": max(
                    0,
                    getattr(req, "num_prefill_tokens", 0) - getattr(req, "num_processed_tokens", 0),
                ),
                "replica": self._replica_id,
            }
            self._batch_request_details_rows.append(row)

            # now record that this request has been scheduled at least once
            self._ever_scheduled_req_ids.add(req.id)





    @if_write_metrics
    def on_request_arrival(self, request: Request) -> None:
        if not self._config.store_request_metrics:
            return

        self._request_metrics_time_series[RequestMetricsTimeSeries.REQUEST_ARRIVAL].put(
            request.arrived_at, 1
        )
        self._token_metrics_time_series[TokenMetricsTimeSeries.PREFILL_ARRIVAL].put(
            request.arrived_at, request.num_prefill_tokens
        )
        self._token_metrics_time_series[
            TokenMetricsTimeSeries.PREFILL_TOKENS_OUTSTANDING
        ].put(request.arrived_at, request.num_prefill_tokens)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id, request.total_tokens
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].put(request.id, request.num_prefill_tokens)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS
        ].put(request.id, request.num_decode_tokens)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_PD_RATIO].put(
            request.id, request.pd_ratio
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
        ].put(request.id, request.arrived_at - self._last_request_arrived_at)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_ARRIVED_AT].put(
            request.id, request.arrived_at
        )
        self._last_request_arrived_at = request.arrived_at

    @if_write_metrics
    def on_request_end(self, request: Request) -> None:
        if not self._config.store_request_metrics:
            return

        self._request_metrics_time_series[
            RequestMetricsTimeSeries.REQUEST_COMPLETION
        ].put(request.completed_at, 1)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.e2e_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(request.id, request.e2e_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
        ].put(request.id, request.model_execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.model_execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time + request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
        ].put(
            request.id,
            (request.execution_time + request.preempted_time)
            / request.num_decode_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(request.id, request.prefill_completed_at - request.arrived_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.arrived_at)
            / request.num_prefill_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(request.id, request.prefill_completed_at - request.scheduled_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.scheduled_at)
            / request.num_prefill_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.completed_at - request.prefill_completed_at)
            / request.num_decode_tokens,
        )

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_NUM_RESTARTS
        ].put(request.id, request.num_restarts)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS_CACHED
        ].put(request.id, request.num_prefill_tokens_cached)

    def _update_per_token_execution_times(
        self, time: float, request: Request, batch: Batch
    ) -> None:
        if not self._config.store_token_completion_metrics:
            return

        if request.has_started_decode:
            self._token_metrics_time_distribution[
                TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME
            ].put(
                time - batch.scheduled_at + request.latest_iteration_scheduling_delay,
            )
            # TODO(t-nitinkedia): Enable decode completions behind a feature flag because of perf overhead
            # self._token_metrics_time_series[
            #     TokenMetricsTimeSeries.DECODE_COMPLETIONS
            # ].put(time, 1)
        elif time == request.prefill_completed_at:
            # if prefill has just finished in this iteration, update the prefill completion time series
            self._token_metrics_time_series[
                TokenMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put(
                time,
                request.num_prefill_tokens,
            )

    def _push_metric(
        self, metric_name: OperationMetrics, batch_id: int, value: float
    ) -> None:
        if metric_name in OperationMetrics:
            self._operation_metrics[metric_name].put(value)
            if self._config.keep_individual_batch_metrics:
                self._operation_metrics_per_batch[metric_name].put(batch_id, value)
        elif metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name].put(value)
            if self._config.keep_individual_batch_metrics:
                self._cpu_operation_metrics_per_batch[metric_name].put(batch_id, value)
        elif metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name].put(value)
            if self._config.keep_individual_batch_metrics:
                self._batch_metrics_time_distribution_per_batch[metric_name].put(
                    batch_id, value
                )
        elif metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name].put(value)
            if self._config.keep_individual_batch_metrics:
                self._batch_metrics_count_distribution_per_batch[metric_name].put(
                    batch_id, value
                )
        else:
            raise ValueError(f"Invalid metric name {metric_name}")

    @if_write_metrics
    def on_replica_schedule(self, time: float, memory_usage_percent: int) -> None:
        if not self._config.store_utilization_metrics:
            return

        self._replica_memory_usage.put(time, memory_usage_percent)
        if self._config.keep_individual_batch_metrics:
            self._replica_memory_usage_per_batch.put(time, memory_usage_percent)

    @if_write_metrics
    def on_replica_stage_schedule(
        self,
        time: float,
        stage_id: int,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
    ) -> None:
        if self._config.store_utilization_metrics:
            self._replica_busy_time[stage_id - 1].put(time, 100)
            mfu = self._mfu_calculator.get_mfu(batch_stage)
            self._replica_mfu[stage_id - 1].put(time, mfu)

        if not self._config.store_operation_metrics:
            return

        batch_id = batch_stage._batch_id
        self._push_metric(
            OperationMetrics.MLP_UP_PROJ,
            batch_id,
            execution_time.mlp_layer_up_proj_execution_time,
        )
        self._push_metric(
            OperationMetrics.MLP_ACTIVATION,
            batch_id,
            execution_time.mlp_layer_act_execution_time,
        )
        self._push_metric(
            OperationMetrics.MLP_DOWN_PROJ,
            batch_id,
            execution_time.mlp_layer_down_proj_execution_time,
        )
        self._push_metric(
            OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            batch_id,
            execution_time.mlp_all_reduce_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_PRE_PROJ,
            batch_id,
            execution_time.attention_pre_proj_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_POST_PROJ,
            batch_id,
            execution_time.attention_post_proj_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            batch_id,
            execution_time.attention_all_reduce_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_PREFILL,
            batch_id,
            execution_time.attention_prefill_execution_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_DECODE,
            batch_id,
            execution_time.attention_decode_execution_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_KV_CACHE_SAVE,
            batch_id,
            execution_time.attention_kv_cache_save_execution_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_ROPE,
            batch_id,
            execution_time.attention_rope_execution_time,
        )
        self._push_metric(OperationMetrics.ADD, batch_id, execution_time.add_time * 2)
        self._push_metric(
            OperationMetrics.INPUT_LAYERNORM,
            batch_id,
            execution_time.attn_norm_time,
        )
        self._push_metric(
            OperationMetrics.POST_ATTENTION_LAYERNORM,
            batch_id,
            execution_time.mlp_norm_time,
        )

        self._push_metric(
            OperationMetrics.PIPELINE_SEND_RECV,
            batch_id,
            execution_time.pipeline_parallel_communication_time,
        )
        self._push_metric(
            CpuOperationMetrics.SCHEDULE, batch_id, execution_time.schedule_time
        )
        self._push_metric(
            CpuOperationMetrics.SAMPLER_E2E, batch_id, execution_time.sampler_e2e_time
        )
        self._push_metric(
            CpuOperationMetrics.PREPARE_INPUTS_E2E,
            batch_id,
            execution_time.prepare_inputs_e2e_time,
        )
        self._push_metric(
            CpuOperationMetrics.MODEL_EXECUTION_E2E,
            batch_id,
            execution_time.model_time_ms,
        )
        self._push_metric(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS,
            batch_id,
            execution_time.process_model_outputs_time,
        )
        self._push_metric(
            CpuOperationMetrics.RAY_COMM_TIME, batch_id, execution_time.ray_comm_time
        )

    @if_write_metrics
    def on_batch_stage_end(
        self, batch_stage: BatchStage, time: float, stage_id: int
    ) -> None:
        if not self._config.store_utilization_metrics:
            return
        self._replica_busy_time[stage_id - 1].put(time, 0)
        self._replica_mfu[stage_id - 1].put(time, 0)

    @if_write_metrics
    def on_batch_end(
        self, time: float, batch: Batch, memory_usage_percent: int
    ) -> None:
        if self._config.store_utilization_metrics:
            self._replica_memory_usage.put(time, memory_usage_percent)
            if self._config.keep_individual_batch_metrics:
                self._replica_memory_usage_per_batch.put(time, memory_usage_percent)

        for request in batch.requests:
            self._update_per_token_execution_times(time, request, batch)

        # --- NEW: stash scheduler-populated batch attributes into our extra DataSeries
        # We rely on the scheduler to annotate the Batch object:
        #   batch.kv_free_blocks_before
        #   batch.kv_free_blocks_after
        #   batch.request_ids_in_batch  (semicolon-joined)
        if self._config.store_batch_metrics:
            if hasattr(batch, "kv_free_blocks_before") and batch.kv_free_blocks_before is not None:
                self._batch_extras_per_batch["kv_free_blocks_before"].put(batch.id, batch.kv_free_blocks_before)

            if hasattr(batch, "kv_free_blocks_after") and batch.kv_free_blocks_after is not None:
                self._batch_extras_per_batch["kv_free_blocks_after"].put(batch.id, batch.kv_free_blocks_after)

            if hasattr(batch, "request_ids_in_batch") and batch.request_ids_in_batch is not None:
                self._batch_extras_per_batch["batch_request_ids"].put(batch.id, batch.request_ids_in_batch)

            if hasattr(batch, "num_requests_not_selected"):
                self._batch_extras_per_batch["num_requests_not_selected"].put(batch.id, batch.num_requests_not_selected)
            if hasattr(batch, "num_requests_initiated_not_completed"):
                self._batch_extras_per_batch["num_requests_initiated_not_completed"].put(batch.id, batch.num_requests_initiated_not_completed)
            if hasattr(batch, "num_decode_phase_total"):
                self._batch_extras_per_batch["num_decode_phase_total"].put(batch.id, batch.num_decode_phase_total)
            if hasattr(batch, "num_prefill_queue_total"):
                self._batch_extras_per_batch["num_prefill_queue_total"].put(batch.id, batch.num_prefill_queue_total)
            if hasattr(batch, "num_not_initiated_in_queue"):
                self._batch_extras_per_batch["num_not_initiated_in_queue"].put(batch.id, batch.num_not_initiated_in_queue)
            if hasattr(batch, "total_requests_batch_start"):
                self._batch_extras_per_batch["total_requests_batch_start"].put(batch.id, batch.total_requests_batch_start)

        # --- NEW: collect per-request details rows if the scheduler attached them
        # if (
        #     self._config.store_batch_metrics
        #     and hasattr(batch, "_request_details_rows")
        #     and batch._request_details_rows
        # ):
        #     # tag replica and stash
        #     for row in batch._request_details_rows:
        #         row["replica"] = self._replica_id
        #     self._batch_request_details_rows.extend(batch._request_details_rows)


        # #  --- New: function call to make accumulate the rows :
        self.add_batch_request_details(batch)



        if not self._config.store_batch_metrics:
            return

        self._push_metric(
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME,
            batch.id,
            time - batch.scheduled_at,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS,
            batch.id,
            batch.total_num_tokens,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS,
            batch.id,
            batch.num_prefill_tokens,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS,
            batch.id,
            batch.num_decode_tokens,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_SIZE, batch.id, batch.size
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_COMPLETED_PREFILLS,
            batch.id,
            batch.num_completed_prefills,
        )
        self._token_metrics_time_series[
            TokenMetricsTimeSeries.PREFILL_TOKENS_OUTSTANDING
        ].put(time, -1 * batch.num_prefill_tokens)
        # self._push_metric(
        #     BatchMetricsCountDistribution.BATCH_DECODE_CONTEXT_SUM,
        #     batch.id,
        #     batch.decode_context_sum,
        # )
        # self._push_metric(
        #     BatchMetricsCountDistribution.BATCH_DECODE_CONTEXT_SPREAD,
        #     batch.id,
        #     batch.decode_context_spread,
        # )
        # self._push_metric(
        #     BatchMetricsCountDistribution.BATCH_DECODE_CONTEXT_IQR,
        #     batch.id,
        #     batch.decode_context_iqr,
        # )
        # if self._config.keep_individual_batch_metrics:
        #     self._batch_metrics_time_series[
        #         BatchMetricsTimeSeries.BATCH_DECODE_CONTEXT_SUM_TIMESERIES
        #     ].put(time, batch.decode_context_sum)
        #     self._batch_metrics_time_series[
        #         BatchMetricsTimeSeries.BATCH_DECODE_CONTEXT_SPREAD_TIMESERIES
        #     ].put(time, batch.decode_context_spread)
        #     self._batch_metrics_time_series[
        #         BatchMetricsTimeSeries.BATCH_DECODE_CONTEXT_IQR_TIMESERIES
        #     ].put(time, batch.decode_context_iqr)
        #     self._batch_metrics_time_series[
        #         BatchMetricsTimeSeries.BATCH_NUM_PREFILL_TOKENS_TIMESERIES
        #     ].put(time, batch.num_prefill_tokens)

    def store_metrics(self, base_plot_path: str, sim_time: float):
        if self._config.store_request_metrics:
            for dataseries in self._request_metrics_histogram.values():
                dataseries.plot_histogram(
                    base_plot_path, dataseries._y_name, bin_count=25
                )

            for dataseries in self._request_metrics_time_distributions.values():
                dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)
                dataseries.plot_scatter(
                    base_plot_path, f"{dataseries._y_name}_scatter", TIME_STR
                )

            for dataseries in self._request_metrics_time_series.values():
                dataseries.plot_step(
                    base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                )

        if self._config.store_token_completion_metrics:
            for dataseries in self._token_metrics_time_series.values():
                dataseries.plot_step(
                    base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                )

    # def get_merged_df(
    #     self,
    #     dataseries_list: List[DataSeries],
    #     key_to_join: str,
    # ):
    #     dfs = [dataseries.to_df() for dataseries in dataseries_list]
    #     assert all([df[key_to_join].is_unique for df in dfs])
    #     # assert all([len(df) == len(dfs[0]) for df in dfs])

    #     # https://stackoverflow.com/questions/53645882/pandas-merging-101/65167327#65167327
    #     merged_df = pd.concat(
    #         [df.set_index(key_to_join) for df in dfs], axis=1, join="inner"
    #     ).reset_index()

    #     merged_df["replica"] = self._replica_id
    #     return merged_df

    def get_merged_df(self, dataseries_list: List[DataSeries], key_to_join: str):
        # Build frames and drop empties
        frames = []
        for ds in dataseries_list:
            df = ds.to_df()
            if df is None or df.empty:
                continue
            # ensure uniqueness for non-empty frames
            assert df[key_to_join].is_unique
            frames.append(df.set_index(key_to_join))

        if not frames:
            # No data yet -> return an empty shell with the join key + replica
            return pd.DataFrame(columns=[key_to_join, "replica"])

        # Use OUTER join so missing series don't drop batches
        merged_df = pd.concat(frames, axis=1, join="outer").reset_index()
        merged_df["replica"] = self._replica_id
        return merged_df




    def get_request_metrics_df(self):
        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())
        return self.get_merged_df(all_request_metrics, REQUEST_ID_STR)

    def get_batch_metrics_df(self):
        # Assembing the newer columns here 
        all_batch_metrics = list(
            self._batch_metrics_count_distribution_per_batch.values()
        ) + list(self._batch_metrics_time_distribution_per_batch.values()) \
        + list(self._batch_extras_per_batch.values())  # <-- NEW
        return self.get_merged_df(all_batch_metrics, BATCH_ID_STR)
        # all_batch_metrics = list(
        #     self._batch_metrics_count_distribution_per_batch.values()
        # ) + list(self._batch_metrics_time_distribution_per_batch.values())
        # return self.get_merged_df(all_batch_metrics, BATCH_ID_STR)

    ## --> New : to create our new csv file : batch_request.csv 
    def get_batch_request_details_df(self) -> pd.DataFrame:
        cols = [
            "batch_id","request_id","slo_type","slo_remaining_ms",
            "only_in_queue","added_in_last_batch","request_type_after_batch",
            "kv_context_len_tokens","prefill_len_tokens","prefill_remaining_tokens",
            "replica",
        ]
        if not self._batch_request_details_rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self._batch_request_details_rows)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]



    def get_operation_metrics_df(self):
        all_operation_metrics = list(self._operation_metrics_per_batch.values()) + list(
            self._cpu_operation_metrics_per_batch.values()
        )
        return self.get_merged_df(all_operation_metrics, BATCH_ID_STR)

    def get_replica_busy_time(self):
        replica_busy_time_dict = {}
        for stage_idx in range(self._num_pipeline_stages):
            replica_busy_time_dict[stage_idx] = self._replica_busy_time[
                stage_idx
            ].get_stats("replica_busy_time")
        return replica_busy_time_dict

    def get_replica_mfu(self):
        replica_mfu_dict = {}
        for stage_idx in range(self._num_pipeline_stages):
            replica_mfu_dict[stage_idx] = self._replica_mfu[stage_idx].get_stats(
                "replica_mfu"
            )
        return replica_mfu_dict

    # def get_prefix_cache_stats(self):
    #     cached_tokens_stats = self._request_metrics_histogram[
    #         RequestMetricsHistogram.REQUEST_PREFILL_TOKENS_CACHED
    #     ].get_stats()
    #     total_tokens_stats = self._request_metrics_histogram[
    #         RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
    #     ].get_stats()
    #     return {
    #         "cached_tokens_sum": int(cached_tokens_stats["sum"]),
    #         "total_tokens_sum": int(total_tokens_stats["sum"]),
    #         "hit_ratio": cached_tokens_stats["sum"] / total_tokens_stats["sum"],
    #     }


    def get_prefix_cache_stats(self):
        cached_stats = self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS_CACHED
        ].get_stats() or {}
        total_stats = self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].get_stats() or {}

        cached_sum = int(cached_stats.get("sum", 0) or 0)
        total_sum  = int(total_stats.get("sum", 0) or 0)
        hit_ratio = (cached_sum / total_sum) if total_sum > 0 else 0.0

        return {
            "cached_tokens_sum": cached_sum,
            "total_tokens_sum": total_sum,
            "hit_ratio": hit_ratio,
        }