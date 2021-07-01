# CI testing script

EXIT_CODE=0

root_dir=$(pwd)
test_dir=$(pwd)/tests

pip install wheel
pip install -r tests/requirements.txt
pip install .
pytest -s -v tests/
local_exit_code=$?

if [[ ! $local_exit_code == 0 ]]; then
  EXIT_CODE=$local_exit_code
  echo $test_dir failed. local_exit_code = $local_exit_code, exit = $EXIT_CODE
fi
cd $root_dir

echo final exit code = $EXIT_CODE
exit $EXIT_CODE