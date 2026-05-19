import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

API_TOKEN_ENV_NAME = "API_TOKEN"

bearer_scheme = HTTPBearer(auto_error=False)


def require_static_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> None:
    expected_token = os.getenv(API_TOKEN_ENV_NAME)
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Токен API не настроен на сервере",
        )

    if credentials is None or not secrets.compare_digest(credentials.credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный или отсутствующий токен авторизации",
            headers={"WWW-Authenticate": "Bearer"},
        )
