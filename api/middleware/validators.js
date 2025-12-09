import { body, param, query, validationResult } from 'express-validator';

/**
 * Input Validation Middleware
 * Protects against injection attacks and ensures data integrity
 */

/**
 * Handle validation errors
 * Returns 400 with error details if validation fails
 */
export const handleValidationErrors = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            error: 'Validation failed',
            details: errors.array().map(err => ({
                field: err.path,
                message: err.msg,
                value: err.value
            }))
        });
    }
    next();
};

/**
 * Registration validation rules
 */
export const validateRegistration = [
    body('email')
        .isEmail()
        .withMessage('Please provide a valid email address')
        .normalizeEmail()
        .isLength({ max: 255 })
        .withMessage('Email must be less than 255 characters'),

    body('password')
        .isLength({ min: 8 })
        .withMessage('Password must be at least 8 characters long')
        .matches(/[a-z]/)
        .withMessage('Password must contain at least one lowercase letter')
        .matches(/[A-Z]/)
        .withMessage('Password must contain at least one uppercase letter')
        .matches(/[0-9]/)
        .withMessage('Password must contain at least one number'),

    body('username')
        .isLength({ min: 3, max: 30 })
        .withMessage('Username must be between 3 and 30 characters')
        .matches(/^[a-zA-Z0-9_]+$/)
        .withMessage('Username can only contain letters, numbers, and underscores')
        .trim()
        .escape(),

    body('gender')
        .optional()
        .isIn(['male', 'female', 'other', 'prefer_not_to_say'])
        .withMessage('Invalid gender option'),

    body('profileImage')
        .optional()
        .isURL()
        .withMessage('Profile image must be a valid URL'),

    handleValidationErrors
];

/**
 * Login validation rules
 */
export const validateLogin = [
    body('email')
        .isEmail()
        .withMessage('Please provide a valid email address')
        .normalizeEmail(),

    body('password')
        .notEmpty()
        .withMessage('Password is required'),

    handleValidationErrors
];

/**
 * Clothing item validation rules
 */
export const validateClothingItem = [
    body('type')
        .notEmpty()
        .withMessage('Clothing type is required')
        .isLength({ max: 100 })
        .withMessage('Type must be less than 100 characters')
        .trim()
        .escape(),

    body('color')
        .optional()
        .isLength({ max: 50 })
        .withMessage('Color must be less than 50 characters')
        .trim(),

    body('style')
        .optional()
        .isIn(['Casual', 'Formal', 'Sport', 'Streetwear', 'Beach', 'Elegant', 'Business'])
        .withMessage('Invalid style option'),

    body('season')
        .optional()
        .isIn(['Spring', 'Summer', 'Fall', 'Winter', 'All Seasons'])
        .withMessage('Invalid season option'),

    body('description')
        .optional()
        .isLength({ max: 500 })
        .withMessage('Description must be less than 500 characters')
        .trim(),

    body('imageUrl')
        .optional()
        .isURL()
        .withMessage('Image URL must be a valid URL'),

    body('price')
        .optional()
        .isFloat({ min: 0, max: 100000 })
        .withMessage('Price must be a positive number'),

    handleValidationErrors
];

/**
 * Outfit validation rules
 */
export const validateOutfit = [
    body('date')
        .optional()
        .isISO8601()
        .withMessage('Date must be a valid ISO date'),

    body('items')
        .isArray({ min: 1 })
        .withMessage('At least one item is required'),

    body('items.*.type')
        .optional()
        .isString()
        .trim(),

    body('items.*.image')
        .optional()
        .isURL()
        .withMessage('Item image must be a valid URL'),

    body('caption')
        .optional()
        .isLength({ max: 500 })
        .withMessage('Caption must be less than 500 characters')
        .trim(),

    body('occasion')
        .optional()
        .isLength({ max: 100 })
        .withMessage('Occasion must be less than 100 characters')
        .trim(),

    handleValidationErrors
];

/**
 * AI chat validation
 */
export const validateAIChat = [
    body('query')
        .notEmpty()
        .withMessage('Query is required')
        .isLength({ max: 1000 })
        .withMessage('Query must be less than 1000 characters')
        .trim(),

    handleValidationErrors
];

/**
 * Image data validation
 */
export const validateImageData = [
    body('imageBase64')
        .notEmpty()
        .withMessage('Image data is required')
        .isLength({ max: 10 * 1024 * 1024 }) // 10MB limit
        .withMessage('Image data too large'),

    handleValidationErrors
];

/**
 * MongoDB ObjectId validation
 */
export const validateObjectId = (paramName = 'id') => [
    param(paramName)
        .isMongoId()
        .withMessage(`Invalid ${paramName} format`),

    handleValidationErrors
];

/**
 * Sanitize a string to prevent XSS
 */
export const sanitizeString = (value) => {
    if (typeof value !== 'string') return value;
    return value
        .replace(/[<>]/g, '') // Remove angle brackets
        .trim();
};
