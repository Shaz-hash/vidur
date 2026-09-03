#include "gv4/features.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace gv4 {

void bind_feature_types(py::module_& module) {
    py::class_<FeatureLayout>(module, "FeatureLayout")
        .def_readonly("schema_version", &FeatureLayout::schema_version)
        .def_readonly("pipeline_stage_count", &FeatureLayout::pipeline_stage_count)
        .def_readonly("global_names", &FeatureLayout::global_names)
        .def_readonly("request_names", &FeatureLayout::request_names)
        .def_readonly("launch_names", &FeatureLayout::launch_names)
        .def_readonly("replica_names", &FeatureLayout::replica_names)
        .def_readonly("microbatch_names", &FeatureLayout::microbatch_names)
        .def_readonly("controller_header_names", &FeatureLayout::controller_header_names)
        .def_readonly("controller_request_names", &FeatureLayout::controller_request_names)
        .def_readonly("adversary_header_names", &FeatureLayout::adversary_header_names)
        .def_readonly("adversary_request_names", &FeatureLayout::adversary_request_names);
    py::class_<FeatureScales>(module, "FeatureScales")
        .def_readonly("window_request_cap", &FeatureScales::window_request_cap)
        .def_readonly("window_prefill_cap", &FeatureScales::window_prefill_cap)
        .def_readonly("active_request_scale", &FeatureScales::active_request_scale)
        .def_readonly("system_prefill_scale", &FeatureScales::system_prefill_scale)
        .def_readonly("system_decode_scale", &FeatureScales::system_decode_scale)
        .def_readonly("request_prefill_scale", &FeatureScales::request_prefill_scale)
        .def_readonly("request_decode_scale", &FeatureScales::request_decode_scale)
        .def_readonly("decode_credit_scale", &FeatureScales::decode_credit_scale)
        .def_readonly("adversary_time_scale", &FeatureScales::adversary_time_scale)
        .def_readonly("launch_age_scale", &FeatureScales::launch_age_scale)
        .def_readonly("lateness_scale", &FeatureScales::lateness_scale)
        .def_readonly("block_token_scale", &FeatureScales::block_token_scale)
        .def_readonly(
            "controller_prefill_action_scale",
            &FeatureScales::controller_prefill_action_scale)
        .def_readonly(
            "controller_kv_block_scale",
            &FeatureScales::controller_kv_block_scale)
        .def_readonly("system_logical_blocks", &FeatureScales::system_logical_blocks)
        .def_readonly("system_logical_tokens", &FeatureScales::system_logical_tokens);
    py::class_<FeatureMatrix>(module, "FeatureMatrix")
        .def_readonly("values", &FeatureMatrix::values)
        .def_readonly("rows", &FeatureMatrix::rows)
        .def_readonly("columns", &FeatureMatrix::columns);
    py::class_<StateFeatures>(module, "StateFeatures")
        .def_readonly("schema_version", &StateFeatures::schema_version)
        .def_readonly("config_manifest_sha256", &StateFeatures::config_manifest_sha256)
        .def_readonly("global_features", &StateFeatures::global_features)
        .def_readonly("request_rows", &StateFeatures::request_rows)
        .def_readonly("request_replica_offsets", &StateFeatures::request_replica_offsets)
        .def_readonly("launch_rows", &StateFeatures::launch_rows)
        .def_readonly("replica_rows", &StateFeatures::replica_rows)
        .def_readonly("microbatch_rows", &StateFeatures::microbatch_rows)
        .def_readonly(
            "microbatch_replica_offsets",
            &StateFeatures::microbatch_replica_offsets);
    py::class_<ControllerActionFeatures>(module, "ControllerActionFeatures")
        .def_readonly("header", &ControllerActionFeatures::header)
        .def_readonly(
            "affected_request_rows",
            &ControllerActionFeatures::affected_request_rows);
    py::class_<AdversaryActionFeatures>(module, "AdversaryActionFeatures")
        .def_readonly("header", &AdversaryActionFeatures::header)
        .def_readonly(
            "affected_request_rows",
            &AdversaryActionFeatures::affected_request_rows);
    py::class_<FeatureBuilder>(module, "FeatureBuilder")
        .def(py::init<Config>())
        .def_property_readonly(
            "config",
            &FeatureBuilder::config,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "layout",
            &FeatureBuilder::layout,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "scales",
            &FeatureBuilder::scales,
            py::return_value_policy::reference_internal)
        .def("build_state", &FeatureBuilder::build_state)
        .def("build_controller_action", &FeatureBuilder::build_controller_action)
        .def("build_adversary_action", &FeatureBuilder::build_adversary_action);
}

}  // namespace gv4
