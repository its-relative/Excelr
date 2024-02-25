-- Practice DML Commands

-- Create a table for practice
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department VARCHAR(50),
    salary DECIMAL(10, 2)
);

-- Insert some sample data
INSERT INTO employees (employee_id, first_name, last_name, department, salary) 
VALUES 
(1, 'John', 'Doe', 'IT', 60000.00),
(2, 'Jane', 'Smith', 'HR', 55000.00),
(3, 'Alice', 'Johnson', 'Finance', 62000.00),
(4, 'Bob', 'Williams', 'IT', 58000.00);

-- Select all employees
SELECT * FROM employees;


set sql_safe_updates=0;
-- Update an employee's salary
UPDATE employees
SET salary = 63000.00
WHERE employee_id = 1;

-- fixing error.. for this use set sql_safe_updates=0

-- Delete an employee
DELETE FROM employees
WHERE employee_id = 4;

-- Insert a new employee
INSERT INTO employees (employee_id, first_name, last_name, department, salary) 
VALUES 
(5, 'Emily', 'Davis', 'Marketing', 60000.00);

-- Select all employees after modifications
SELECT * FROM employees;

-- Practice Exercises:
-- 1. Increase the salary of all employees in the IT department by 10%.
UPDATE employees
SET salary = salary * 1.10
WHERE department = 'IT';

-- 2. Change the department of employee with ID 3 to "Operations".
UPDATE employees
SET department = 'Operations'
WHERE employee_id = 3;

-- 3. Delete all employees whose salary is less than 55000.
DELETE FROM employees
WHERE salary < 55000.00;

-- 4. Insert a new employee with an ID of 6, first name "Michael", last name "Brown", department "Finance", and salary 65000.
INSERT INTO employees (employee_id, first_name, last_name, department, salary) 
VALUES 
(6, 'Michael', 'Brown', 'Finance', 65000.00);

-- 5. Select the first name and last name of all employees whose salary is greater than 60000.
SELECT first_name, last_name
FROM employees
WHERE salary > 60000.00;

-- 6. Update the salary of the employee with the highest salary to 70000.
UPDATE employees
SET salary = 70000.00
WHERE salary = (SELECT MAX(salary) FROM employees);
-- ERROR 1093 (HY000): 
-- You can't specify target table 'employees' for update in FROM clause

-- 7. Delete all employees from the HR department.
DELETE FROM employees
WHERE department = 'HR';

-- 8. Insert a new employee with an ID of 7, first name "Sophia", last name "Lee", department "IT", and salary 62000.
INSERT INTO employees (employee_id, first_name, last_name, department, salary) 
VALUES 
(7, 'Sophia', 'Lee', 'IT', 62000.00);
