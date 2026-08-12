#!/usr/bin/env bash
set -Eeuo pipefail

# Render self-contained EC2 user data with the coordinator SSH key embedded.
# The rendered file contains the private key and must not be committed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP="${SCRIPT_DIR}/ec2_spot_worker_user_data.sh"
OUTPUT="${AGZ_USER_DATA_OUTPUT:-/tmp/vidur-agz-spot-worker-user-data.sh}"
KEY_PATH="${AGZ_SSH_PRIVATE_KEY_PATH:-${HOME}/.ssh/shaz-pr1.pem}"
KNOWN_HOSTS_PATH="${AGZ_SSH_KNOWN_HOSTS_PATH:-}"
XL_HOST="${AGZ_XL_HOST:-}"
XL_OUTPUT_ROOT="${AGZ_XL_OUTPUT_ROOT:-}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -n "${XL_HOST}" ]] || die "AGZ_XL_HOST is required"
[[ -n "${XL_OUTPUT_ROOT}" ]] || die "AGZ_XL_OUTPUT_ROOT is required"
[[ -r "${KEY_PATH}" ]] || die "SSH private key is not readable: ${KEY_PATH}"
[[ -r "${BOOTSTRAP}" ]] || die "bootstrap script is not readable: ${BOOTSTRAP}"

key_b64="$(base64 --wrap=0 < "${KEY_PATH}")"
if [[ -n "${KNOWN_HOSTS_PATH}" ]]; then
    [[ -r "${KNOWN_HOSTS_PATH}" ]] || die "known_hosts is not readable: ${KNOWN_HOSTS_PATH}"
    known_hosts_b64="$(base64 --wrap=0 < "${KNOWN_HOSTS_PATH}")"
else
    known_hosts="$(ssh-keyscan -T 10 -t ed25519 "${XL_HOST}" 2>/dev/null)"
    [[ -n "${known_hosts}" ]] || die "could not obtain the coordinator ED25519 host key"
    known_hosts_b64="$(printf '%s\n' "${known_hosts}" | base64 --wrap=0)"
fi

install -m 0600 /dev/null "${OUTPUT}"
{
    printf '#!/usr/bin/env bash\n'
    printf 'export AGZ_XL_HOST=%q\n' "${XL_HOST}"
    printf 'export AGZ_XL_OUTPUT_ROOT=%q\n' "${XL_OUTPUT_ROOT}"
    printf 'export AGZ_SSH_PRIVATE_KEY_B64=%q\n' "${key_b64}"
    printf 'export AGZ_SSH_KNOWN_HOSTS_B64=%q\n' "${known_hosts_b64}"

    while IFS= read -r name; do
        case "${name}" in
            AGZ_XL_HOST|AGZ_XL_OUTPUT_ROOT|AGZ_SSH_*|AGZ_USER_DATA_OUTPUT)
                continue
                ;;
            AGZ_*)
                printf 'export %s=%q\n' "${name}" "${!name}"
                ;;
        esac
    done < <(compgen -A variable | sort)

    sed -n '2,$p' "${BOOTSTRAP}" | sed '/^[[:space:]]*#[^!]/d; /^[[:space:]]*#$/d; /^[[:space:]]*$/d'
} > "${OUTPUT}"

size_bytes="$(wc -c < "${OUTPUT}")"
if (( size_bytes > 16384 )); then
    gzip -9 "${OUTPUT}"
    mv "${OUTPUT}.gz" "${OUTPUT}"
    size_bytes="$(wc -c < "${OUTPUT}")"
fi
if (( size_bytes > 16384 )); then
    rm -f "${OUTPUT}"
    die "compressed user data is ${size_bytes} bytes; EC2 allows at most 16384"
fi

unset key_b64 known_hosts known_hosts_b64
printf 'Rendered %s (%s bytes). This file contains the SSH private key.\n' \
    "${OUTPUT}" "${size_bytes}"
