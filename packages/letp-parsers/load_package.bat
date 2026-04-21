%FWSW_APPS%/uv/bin/uv add --dev twine
%FWSW_APPS%/uv/bin/uv build

%FWSW_APPS%/uv/bin/uv run twine upload --repository gitea C:\Users\ropic\WORK\LETP\dist\letp_parsers-1.0.2-py3-none-any.whl --config-file %USERPROFILE%\.pypirc --cert %FWSW_APPS%/certs/RLS-CA.crt

pause