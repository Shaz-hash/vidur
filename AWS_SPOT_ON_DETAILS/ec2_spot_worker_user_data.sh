#!/usr/bin/env bash
set -Eeuo pipefail

# EC2 user-data bootstrap for a Vidur AlphaGoZero Spot worker. The launch
# template renderer must supply AGZ_XL_* and either inline SSH values or a
# Secrets Manager secret ID.

umask 077

AWS_REGION="${AGZ_AWS_REGION:-us-east-1}"
ECR_REGISTRY="${AGZ_ECR_REGISTRY:-388660028044.dkr.ecr.us-east-1.amazonaws.com}"
ECR_REPOSITORY="${AGZ_ECR_REPOSITORY:-vidur-agz-exp3-worker}"
ECR_IMAGE_DIGEST="${AGZ_ECR_IMAGE_DIGEST:-sha256:7b47d8ad91e1472949f81bec85ac9154dd286ff731679f7e70b55e4ae37ce2e4}"
IMAGE_REF="${ECR_REGISTRY}/${ECR_REPOSITORY}@${ECR_IMAGE_DIGEST}"

XL_HOST="${AGZ_XL_HOST:-}"
XL_SSH_USER="${AGZ_XL_SSH_USER:-ubuntu}"
XL_OUTPUT_ROOT="${AGZ_XL_OUTPUT_ROOT:-}"
SSH_SECRET_ID="${AGZ_SSH_SECRET_ID:-}"
SSH_PRIVATE_KEY_B64="${AGZ_SSH_PRIVATE_KEY_B64:-}"
SSH_KNOWN_HOSTS_B64="${AGZ_SSH_KNOWN_HOSTS_B64:-}"
PARENT_DATASET_REMOTE_DIR="${AGZ_PARENT_DATASET_REMOTE_DIR:-}"
PARENT_STATE_COUNT="${AGZ_PARENT_STATE_COUNT:-0}"
PARENT_ROOT_PLAYER_FILTER="${AGZ_PARENT_ROOT_PLAYER_FILTER:-any}"
BUFFER_THRESHOLD="${AGZ_BUFFER_THRESHOLD:-4000}"
MEMORY_GIB_PER_GAME="${AGZ_RESOURCE_MEMORY_GIB_PER_GAME:-1.0}"
MODEL_FAMILY="${AGZ_MODEL_FAMILY:-dnn}"
VALUE_FEATURE_SCHEMA="${AGZ_VALUE_FEATURE_SCHEMA:-markov_v2}"
POLICY_FEATURE_SCHEMA="${AGZ_POLICY_FEATURE_SCHEMA:-markov_v2}"

STATE_ROOT="/var/lib/vidur-agz"
SIMULATOR_OUTPUT_HOST="${STATE_ROOT}/simulator_output"
SSH_HOST_DIR="${STATE_ROOT}/ssh"
CONFIG_ROOT="/etc/vidur-agz"
HOST_ENV="${CONFIG_ROOT}/host.env"
CONTAINER_ENV="${CONFIG_ROOT}/container.env"
RUNNER="/usr/local/sbin/vidur-agz-run-worker"
SERVICE_NAME="vidur-agz-spot-worker.service"
APP_HOME="/home/ubuntu/vidur-classical-search"
SIMULATOR_OUTPUT_CONTAINER="${APP_HOME}/simulator_output"
PARENT_DATASET_CONTAINER="${SIMULATOR_OUTPUT_CONTAINER}/GV3_Agent/spot_parent_dataset"
PARENT_DATASET_HOST="${SIMULATOR_OUTPUT_HOST}/GV3_Agent/spot_parent_dataset"
MATPLOTLIB_CACHE_HOST="${STATE_ROOT}/matplotlib-cache"

log() {
    printf '[vidur-agz-bootstrap] %s\n' "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

require_value() {
    local name="$1"
    local value="$2"
    [[ -n "${value}" ]] || die "${name} is required"
    [[ "${value}" != __*__ ]] || die "${name} still contains a template token"
}

retry() {
    local attempts="$1"
    local delay="$2"
    shift 2
    local count=1
    until "$@"; do
        if (( count >= attempts )); then
            return 1
        fi
        log "attempt ${count}/${attempts} failed: $*; retrying in ${delay}s" >&2
        sleep "${delay}"
        count=$((count + 1))
    done
}

require_value AGZ_XL_HOST "${XL_HOST}"
require_value AGZ_XL_OUTPUT_ROOT "${XL_OUTPUT_ROOT}"
if [[ -z "${SSH_SECRET_ID}" ]]; then
    require_value AGZ_SSH_PRIVATE_KEY_B64 "${SSH_PRIVATE_KEY_B64}"
    require_value AGZ_SSH_KNOWN_HOSTS_B64 "${SSH_KNOWN_HOSTS_B64}"
fi
[[ "${XL_OUTPUT_ROOT}" == "${SIMULATOR_OUTPUT_CONTAINER}/"* ]] || \
    die "AGZ_XL_OUTPUT_ROOT must be below ${SIMULATOR_OUTPUT_CONTAINER}"
[[ "${ECR_IMAGE_DIGEST}" == sha256:* ]] || die "AGZ_ECR_IMAGE_DIGEST must be a sha256 digest"
[[ "${PARENT_ROOT_PLAYER_FILTER}" =~ ^(controller|adversary|any)$ ]] || \
    die "AGZ_PARENT_ROOT_PLAYER_FILTER must be controller, adversary, or any"
[[ "${PARENT_STATE_COUNT}" =~ ^[0-9]+$ ]] || die "AGZ_PARENT_STATE_COUNT must be a non-negative integer"
[[ "${BUFFER_THRESHOLD}" =~ ^[1-9][0-9]*$ ]] || die "AGZ_BUFFER_THRESHOLD must be a positive integer"
[[ "${MODEL_FAMILY}" == "dnn" ]] || die "Spot DNN workers require AGZ_MODEL_FAMILY=dnn"
[[ "${VALUE_FEATURE_SCHEMA}" == "markov_v2" ]] || \
    die "Spot DNN workers require AGZ_VALUE_FEATURE_SCHEMA=markov_v2"
[[ "${POLICY_FEATURE_SCHEMA}" == "markov_v2" ]] || \
    die "Spot DNN workers require AGZ_POLICY_FEATURE_SCHEMA=markov_v2"
if (( PARENT_STATE_COUNT > 0 )); then
    require_value AGZ_PARENT_DATASET_REMOTE_DIR "${PARENT_DATASET_REMOTE_DIR}"
fi

if [[ "${AGZ_BOOTSTRAP_DRY_RUN:-0}" == "1" ]]; then
    log "dry-run configuration is valid"
    printf 'image_ref=%s\n' "${IMAGE_REF}"
    printf 'xl_host=%s\n' "${XL_HOST}"
    printf 'xl_output_root=%s\n' "${XL_OUTPUT_ROOT}"
    if [[ -n "${SSH_SECRET_ID}" ]]; then
        printf 'ssh_transport=secrets-manager\n'
    else
        printf 'ssh_transport=inline-user-data\n'
    fi
    printf 'buffer_threshold=%s\n' "${BUFFER_THRESHOLD}"
    printf 'memory_gib_per_game=%s\n' "${MEMORY_GIB_PER_GAME}"
    exit 0
fi

[[ "${EUID}" -eq 0 ]] || die "EC2 user data must run as root"

exec > >(tee -a /var/log/vidur-agz-bootstrap.log | logger -t vidur-agz-bootstrap -s 2>/dev/console) 2>&1
trap 'rc=$?; log "bootstrap failed at line ${LINENO} with exit ${rc}"; exit ${rc}' ERR

install_host_dependencies() {
    export DEBIAN_FRONTEND=noninteractive
    retry 8 10 apt-get update
    retry 8 10 apt-get install -y --no-install-recommends \
        ca-certificates curl docker.io jq openssh-client rsync unzip
    systemctl enable --now docker

    if ! command -v aws >/dev/null 2>&1; then
        local machine aws_arch tmpdir
        machine="$(uname -m)"
        case "${machine}" in
            aarch64|arm64) aws_arch="aarch64" ;;
            x86_64|amd64) aws_arch="x86_64" ;;
            *) die "unsupported AWS CLI architecture: ${machine}" ;;
        esac
        tmpdir="$(mktemp -d /tmp/awscliv2.XXXXXX)"
        retry 8 10 curl --fail --location --silent --show-error \
            "https://awscli.amazonaws.com/awscli-exe-linux-${aws_arch}.zip" \
            --output "${tmpdir}/awscliv2.zip"
        unzip -q "${tmpdir}/awscliv2.zip" -d "${tmpdir}"
        "${tmpdir}/aws/install" --update
        rm -rf "${tmpdir}"
    fi
}

read_instance_identity() {
    local token
    token="$(retry 20 2 curl --fail --silent --show-error \
        --request PUT \
        --header 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
        http://169.254.169.254/latest/api/token)"
    INSTANCE_ID="$(retry 20 2 curl --fail --silent --show-error \
        --header "X-aws-ec2-metadata-token: ${token}" \
        http://169.254.169.254/latest/meta-data/instance-id)"
    require_value EC2_INSTANCE_ID "${INSTANCE_ID}"
    WORKER_ID="spot-${INSTANCE_ID}"
    local id_hex
    id_hex="$(printf '%s' "${INSTANCE_ID}" | sha256sum | cut -c1-8)"
    GAME_ID_START=$((100000000 + (16#${id_hex} % 1900000000)))
}

install_ssh_bundle() {
    local secret_json private_key known_hosts

    if [[ -n "${SSH_SECRET_ID}" ]]; then
        secret_json="$(retry 20 5 aws secretsmanager get-secret-value \
            --region "${AWS_REGION}" \
            --secret-id "${SSH_SECRET_ID}" \
            --query SecretString \
            --output text)"
        private_key="$(jq -er '.private_key' <<<"${secret_json}")"
        known_hosts="$(jq -er '.known_hosts' <<<"${secret_json}")"
    else
        private_key="$(base64 --decode <<<"${SSH_PRIVATE_KEY_B64}")"
        known_hosts="$(base64 --decode <<<"${SSH_KNOWN_HOSTS_B64}")"
    fi
    [[ "${private_key}" == *"PRIVATE KEY"* ]] || die "decoded SSH private key is invalid"
    [[ -n "${known_hosts}" ]] || die "decoded coordinator known_hosts is empty"

    install -d -m 0700 -o 1000 -g 1000 "${SSH_HOST_DIR}"
    printf '%s\n' "${private_key}" > "${SSH_HOST_DIR}/id_ed25519"
    printf '%s\n' "${known_hosts}" > "${SSH_HOST_DIR}/known_hosts"
    cat > "${SSH_HOST_DIR}/config" <<EOF
Host agz-coordinator
    HostName ${XL_HOST}
    User ${XL_SSH_USER}
    IdentityFile /home/ubuntu/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking yes
    UserKnownHostsFile /home/ubuntu/.ssh/known_hosts
    ServerAliveInterval 30
    ServerAliveCountMax 6
EOF
    chmod 0600 "${SSH_HOST_DIR}/id_ed25519" "${SSH_HOST_DIR}/config"
    chmod 0644 "${SSH_HOST_DIR}/known_hosts"
    chown -R 1000:1000 "${SSH_HOST_DIR}"
    unset secret_json private_key known_hosts
    unset SSH_PRIVATE_KEY_B64 SSH_KNOWN_HOSTS_B64
    unset AGZ_SSH_PRIVATE_KEY_B64 AGZ_SSH_KNOWN_HOSTS_B64
}

pull_image() {
    retry 20 5 aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${ECR_REGISTRY}"
    retry 8 10 docker pull "${IMAGE_REF}"
    docker logout "${ECR_REGISTRY}" || true
}

write_runtime_configuration() {
    install -d -m 0755 -o 1000 -g 1000 \
        "${STATE_ROOT}" \
        "${SIMULATOR_OUTPUT_HOST}" \
        "${SIMULATOR_OUTPUT_HOST}/GV3_Agent"
    install -d -m 0755 "${CONFIG_ROOT}"
    install -d -m 0755 -o 1000 -g 1000 \
        "${PARENT_DATASET_HOST}" \
        "${MATPLOTLIB_CACHE_HOST}"

    {
        printf 'IMAGE_REF=%q\n' "${IMAGE_REF}"
        printf 'WORKER_ID=%q\n' "${WORKER_ID}"
        printf 'GAME_ID_START=%q\n' "${GAME_ID_START}"
        printf 'XL_OUTPUT_ROOT=%q\n' "${XL_OUTPUT_ROOT}"
        printf 'BUFFER_THRESHOLD=%q\n' "${BUFFER_THRESHOLD}"
        printf 'MEMORY_GIB_PER_GAME=%q\n' "${MEMORY_GIB_PER_GAME}"
        printf 'PARENT_STATE_COUNT=%q\n' "${PARENT_STATE_COUNT}"
        printf 'PARENT_ROOT_PLAYER_FILTER=%q\n' "${PARENT_ROOT_PLAYER_FILTER}"
        printf 'SIMULATOR_OUTPUT_HOST=%q\n' "${SIMULATOR_OUTPUT_HOST}"
        printf 'MATPLOTLIB_CACHE_HOST=%q\n' "${MATPLOTLIB_CACHE_HOST}"
        printf 'SSH_HOST_DIR=%q\n' "${SSH_HOST_DIR}"
    } > "${HOST_ENV}"
    chmod 0600 "${HOST_ENV}"

    : > "${CONTAINER_ENV}"
    printf 'AGZ_MODEL_FAMILY=%s\n' "${MODEL_FAMILY}" >> "${CONTAINER_ENV}"
    printf 'AGZ_VALUE_FEATURE_SCHEMA=%s\n' "${VALUE_FEATURE_SCHEMA}" >> "${CONTAINER_ENV}"
    printf 'AGZ_POLICY_FEATURE_SCHEMA=%s\n' "${POLICY_FEATURE_SCHEMA}" >> "${CONTAINER_ENV}"
    local name
    while IFS= read -r name; do
        case "${name}" in
            AGZ_AWS_REGION|AGZ_BOOTSTRAP_DRY_RUN|AGZ_ECR_*|AGZ_PARENT_DATASET_REMOTE_DIR|AGZ_SSH_*|AGZ_XL_HOST|AGZ_XL_SSH_USER)
                continue
                ;;
            AGZ_*|OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|NUMEXPR_NUM_THREADS)
                case "${name}" in
                    AGZ_MODEL_FAMILY|AGZ_VALUE_FEATURE_SCHEMA|AGZ_POLICY_FEATURE_SCHEMA)
                        continue
                        ;;
                esac
                printf '%s=%s\n' "${name}" "${!name}" >> "${CONTAINER_ENV}"
                ;;
        esac
    done < <(compgen -A variable | sort)
    grep -q '^OMP_NUM_THREADS=' "${CONTAINER_ENV}" || printf 'OMP_NUM_THREADS=1\n' >> "${CONTAINER_ENV}"
    grep -q '^OPENBLAS_NUM_THREADS=' "${CONTAINER_ENV}" || printf 'OPENBLAS_NUM_THREADS=1\n' >> "${CONTAINER_ENV}"
    grep -q '^MKL_NUM_THREADS=' "${CONTAINER_ENV}" || printf 'MKL_NUM_THREADS=1\n' >> "${CONTAINER_ENV}"
    grep -q '^NUMEXPR_NUM_THREADS=' "${CONTAINER_ENV}" || printf 'NUMEXPR_NUM_THREADS=1\n' >> "${CONTAINER_ENV}"
    chmod 0600 "${CONTAINER_ENV}"
}

stage_parent_dataset() {
    [[ -n "${PARENT_DATASET_REMOTE_DIR}" ]] || return 0
    log "staging parent dataset from coordinator"
    retry 8 15 docker run --rm --network host \
        -v "${SSH_HOST_DIR}:/home/ubuntu/.ssh:ro" \
        -v "${PARENT_DATASET_HOST}:${PARENT_DATASET_CONTAINER}" \
        "${IMAGE_REF}" shell -lc \
        "rsync -az --partial --delay-updates --timeout=120 agz-coordinator:${PARENT_DATASET_REMOTE_DIR%/}/ ${PARENT_DATASET_CONTAINER}/"
}

prewarm_matplotlib_cache() {
    log "prewarming shared Matplotlib font cache"
    retry 3 5 docker run --rm \
        -v "${MATPLOTLIB_CACHE_HOST}:/home/ubuntu/.cache/matplotlib" \
        "${IMAGE_REF}" shell -lc \
        '/home/ubuntu/vidur-classical-search/.venv/bin/python3 -c "import matplotlib.font_manager"'
}

verify_coordinator_connection() {
    retry 20 5 docker run --rm --network host \
        -v "${SSH_HOST_DIR}:/home/ubuntu/.ssh:ro" \
        "${IMAGE_REF}" shell -lc \
        'ssh -o BatchMode=yes -o ConnectTimeout=15 agz-coordinator true'
}

install_runner() {
    cat > "${RUNNER}" <<'RUNNER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
source /etc/vidur-agz/host.env

APP_HOME=/home/ubuntu/vidur-classical-search
SIMULATOR_OUTPUT_CONTAINER=${APP_HOME}/simulator_output
WORKER_OUTPUT_ROOT=${XL_OUTPUT_ROOT}/spot_workers/${WORKER_ID}
PARENT_ARGS=()
if [[ "${PARENT_STATE_COUNT}" -gt 0 ]]; then
    PARENT_ARGS=(
        --parent-dataset-dir "${SIMULATOR_OUTPUT_CONTAINER}/GV3_Agent/spot_parent_dataset"
        --parent-state-count "${PARENT_STATE_COUNT}"
        --parent-root-player-filter "${PARENT_ROOT_PLAYER_FILTER}"
    )
fi

exec /usr/bin/docker run --rm \
    --name vidur-agz-spot-worker \
    --network host \
    --stop-timeout 240 \
    --env-file /etc/vidur-agz/container.env \
    -v "${SIMULATOR_OUTPUT_HOST}:${SIMULATOR_OUTPUT_CONTAINER}" \
    -v "${SSH_HOST_DIR}:/home/ubuntu/.ssh:ro" \
    -v "${MATPLOTLIB_CACHE_HOST}:/home/ubuntu/.cache/matplotlib" \
    "${IMAGE_REF}" \
    spot-worker \
    --worker-id "${WORKER_ID}" \
    --run-id spot-fleet \
    --output-root "${WORKER_OUTPUT_ROOT}" \
    --xl-host agz-coordinator \
    --xl-output-root "${XL_OUTPUT_ROOT}" \
    --controller-repo "${APP_HOME}" \
    --controller-python "${APP_HOME}/.venv/bin/python3" \
    --game-id-start "${GAME_ID_START}" \
    --parallel-games 0 \
    --buffer-threshold "${BUFFER_THRESHOLD}" \
    --assignment-lease-sec 300 \
    --heartbeat-sec 10 \
    --resource-reserve-cpus -1 \
    --resource-memory-gib-per-game "${MEMORY_GIB_PER_GAME}" \
    "${PARENT_ARGS[@]}"
RUNNER_EOF
    chmod 0755 "${RUNNER}"

    cat > "/etc/systemd/system/${SERVICE_NAME}" <<EOF
[Unit]
Description=Vidur AlphaGoZero Spot worker
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f vidur-agz-spot-worker
ExecStart=${RUNNER}
Restart=always
RestartSec=15
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now "${SERVICE_NAME}"
}

log "starting bootstrap for ${IMAGE_REF}"
install_host_dependencies
read_instance_identity
log "instance=${INSTANCE_ID} worker=${WORKER_ID} game_id_start=${GAME_ID_START}"
install_ssh_bundle
pull_image
write_runtime_configuration
verify_coordinator_connection
stage_parent_dataset
prewarm_matplotlib_cache
install_runner

systemctl --no-pager --full status "${SERVICE_NAME}" || true
log "bootstrap completed successfully"
