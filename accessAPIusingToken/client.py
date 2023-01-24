import logging
import os
import json
import msal


class ClientBase:
    prefix: str = "Bearer"
    auth_header_name: str = "Authorization"

    def __init__(self, endpoint=None):
        super(ClientBase, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.scope = ["xyz"]
        self.client_id = 'xyz'
        self.authority = 'https://login.microsoftonline.com/9485acfb-a348-4a74-8408-be47f710df4b'
        self.client_credential = "AnupamBose"
        # find appropriate user settings
        user_settings_path = os.path.join(
            os.path.expanduser('~'),
            'anupam.json'
        )
        if endpoint is None:
            with open(user_settings_path, 'r') as f:
                settings = json.load(f)
            self._endpoint_base = settings['user_mgmt_endpoint']
        else:
            self._endpoint_base = endpoint
        self.app = msal.ConfidentialClientApplication(
         client_id=self.client_id,
         authority=self.authority,
         client_credential=self.client_credential)
        self.authResult = self.app.acquire_token_for_client(scopes=self.scope)
        self._headers: dict = {self.auth_header_name: f"{self.prefix} {self.authResult['access_token']}"} if self.prefix else self.authResult['access_token']
        """Get headers to be used in authenticated endpoints"""

    def _post_response_check(self, response, method):
        if len(response.text) > 0:
            self.logger.warning(
                f'{method} add sent a response\n{response.text}\n'
            )
