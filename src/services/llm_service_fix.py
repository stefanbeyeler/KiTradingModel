    async def check_model_available(self) -> bool:
        """Check if the configured model is available in Ollama."""
        try:
            client = self._get_client()
            models = client.list()

            # Handle different Ollama API response formats
            available_models = []
            models_data = models.get("models", []) if isinstance(models, dict) else models

            for m in models_data:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or str(m)
                else:
                    name = getattr(m, 'model', None) or getattr(m, 'name', None) or str(m)
                available_models.append(name)

            is_available = any(self.model in m for m in available_models)

            if not is_available:
                logger.warning(
                    f"Model {self.model} not found. Available models: {available_models}"
                )

            return is_available
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
