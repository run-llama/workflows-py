# initialize pypi_token
pypi_token="no_token"
package=""

while
    [[ $# -gt 0 ]] \
        ;
do
    case "$1" in
    -t | --token)
        pypi_token="$2"
        shift 2
        ;;
    --package)
        package="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if [[ $pypi_token == "no_token" ]]; then
    if [[ $PYPI_TOKEN == "" ]]; then
        echo "No token provided and no token in the environment, exiting..."
        exit 1
    else
        pypi_token="$PYPI_TOKEN"
    fi
fi

if [[ -z $package ]] \
    ;
then
    exit 1
fi

# check pre-release
uv venv && source .venv/bin/activate && uv pip install -r scripts/requirements.txt
do_not_release=$(python3 scripts/pre_release_check.py --package $package)

if [ "$do_not_release" -eq 1 ]; then
   echo "Nothing to release"
   exit 0
else
    # build and publish llama_cloud_services
    ## build
    cd packages/${package}
    uv build
    ## publish
    uv publish --token $pypi_token
fi
