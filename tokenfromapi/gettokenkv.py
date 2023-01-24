# Given the client ID and tenant ID for an app registered in Azure,
# provide an Azure AD access token and a refresh token.

# If the caller is not already signed in to Azure, the caller's
# web browser will prompt the caller to sign in first.

import sys
import os

# pip install msal
from msal import PublicClientApplication
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential


class GetToken:
    def __init__(self):
        # You can hard-code the registered app's client ID and tenant ID here,
        # or you can provide them as command-line arguments to this script.
        self.client_id = "anupam"
        self.tenant_id = "xyz"
        self.client_secret = "xyz."

        # Do not modify this variable. It represents the programmatic ID for
        # Azure Databricks along with the default scope of '/.default'.
        self.scopes = 'xyz'
        # Check for too few or too many command-line arguments.
        if (len(sys.argv) > 1) and (len(sys.argv) != 5):
            print("Usage: get-tokens.py <client ID> <tenant ID>")
            exit(1)

        # If the registered app's client ID and tenant ID are provided as
        # command-line variables, set them here.
        if len(sys.argv) > 1:
            self.client_id = sys.argv[1]
            self.tenant_id = sys.argv[2]
            self.username = sys.argv[2]
            self.password = sys.argv[2]

    def acquire_token(self):
        app = PublicClientApplication(client_id=self.client_id, authority="https://login.microsoftonline.com/" + self.tenant_id)
        # get username and password from Key vault
        _credential = ClientSecretCredential(
         tenant_id=self.tenant_id,
         client_id=self.client_id,
         client_secret=self.client_secret
        )
        keyVaultName = os.getenv("KEY_VAULT_NAME")
        KVUri = f"https://{keyVaultName}.vault.azure.net"
        client = SecretClient(vault_url=KVUri, credential=_credential)
        acquire_tokens_result = app.acquire_token_by_username_password(
            username=client.get_secret('TestUserName').value,
            password=client.get_secret('TestPassword').value,
            scopes=self.scopes)
        if 'error' in acquire_tokens_result:
            return print("Error: " + acquire_tokens_result['error'] + "Description: " + acquire_tokens_result['error_description'])
        else:
            return acquire_tokens_result['access_token']

a = GetToken()
a.acquire_token()