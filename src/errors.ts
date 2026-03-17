/**
 * Base class for configuration errors.
 * Extends Error so it won't be caught by generic error handlers
 * that specifically check for non-Error instances.
 */
export class ConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigurationError";
  }
}
