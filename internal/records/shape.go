package records

import (
	"database/sql"
	"fmt"
	"strings"
	"sync"
)

// shape is what one records schema REQUIRES of a database: its tables and each table's
// columns, then the indexes and triggers over them. It is read from the authored DDL, and a
// database satisfies it when every named object and column exists. Extra tables, columns,
// indexes and triggers, and any difference in the stored DDL text, are the database's own.
type shape struct {
	tables  []requiredTable
	objects []requiredObject // indexes and triggers, created after every table has its columns
}

type requiredTable struct {
	name    string
	ddl     string
	columns []requiredColumn
}

type requiredColumn struct {
	name string
	// add is the ADD COLUMN definition that restores the column, or "" when SQLite
	// cannot add it to an existing table (a primary-key column).
	add string
}

type requiredObject struct {
	kind, name, ddl string
}

type schemaDB interface {
	Exec(string, ...any) (sql.Result, error)
	Query(string, ...any) (*sql.Rows, error)
}

var currentShape struct {
	sync.Once
	shape shape
	err   error
}

func requiredCurrentShape() (shape, error) {
	currentShape.Do(func() { currentShape.shape, currentShape.err = shapeOf(schema) })
	return currentShape.shape, currentShape.err
}

// shapeOf executes the authored statements in memory and reads back what they created.
func shapeOf(statements []string) (shape, error) {
	db, err := sql.Open("sqlite", ":memory:"+pragmas)
	if err != nil {
		return shape{}, err
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	for _, statement := range statements {
		if _, err := db.Exec(statement); err != nil {
			return shape{}, err
		}
	}
	rows, err := db.Query(`SELECT type,name,COALESCE(sql,'') FROM sqlite_master
		WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY rowid`)
	if err != nil {
		return shape{}, err
	}
	var out shape
	for rows.Next() {
		var object requiredObject
		if err := rows.Scan(&object.kind, &object.name, &object.ddl); err != nil {
			rows.Close()
			return shape{}, err
		}
		if object.kind == "table" {
			out.tables = append(out.tables, requiredTable{name: object.name, ddl: object.ddl})
			continue
		}
		out.objects = append(out.objects, object)
	}
	if err := rows.Close(); err != nil {
		return shape{}, err
	}
	for index := range out.tables {
		columns, err := tableColumns(db, out.tables[index].name)
		if err != nil {
			return shape{}, err
		}
		out.tables[index].columns = columns
	}
	return out, nil
}

// tableColumns reads a table's columns; an absent table has none.
func tableColumns(db schemaDB, table string) ([]requiredColumn, error) {
	rows, err := db.Query(`SELECT name,type,"notnull",dflt_value,pk FROM pragma_table_info(?)`, table)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []requiredColumn
	for rows.Next() {
		var name, kind string
		var notNull, primary int
		var fallback sql.NullString
		if err := rows.Scan(&name, &kind, &notNull, &fallback, &primary); err != nil {
			return nil, err
		}
		out = append(out, requiredColumn{name: name, add: addColumn(name, kind, notNull != 0, fallback, primary != 0)})
	}
	return out, rows.Err()
}

// addColumn restores a missing column with its declared type and default. A NOT NULL
// column with no default gets its type's zero value, which is what every migration
// that introduced such a column backfilled.
func addColumn(name, kind string, notNull bool, fallback sql.NullString, primary bool) string {
	if primary {
		return ""
	}
	out := fmt.Sprintf("%q %s", name, kind)
	value := fallback.String
	if notNull && !fallback.Valid {
		switch upper := strings.ToUpper(kind); {
		case strings.Contains(upper, "INT"), strings.Contains(upper, "REAL"),
			strings.Contains(upper, "FLOA"), strings.Contains(upper, "DOUB"):
			value = "0"
		case strings.Contains(upper, "BLOB"):
			value = "x''"
		default:
			value = "''"
		}
	}
	if value != "" {
		out += " DEFAULT " + value
	}
	if notNull {
		out += " NOT NULL"
	}
	return out
}

// conform brings db up to the required shape: a missing table, column, index or trigger is
// created. Nothing present is altered or dropped, so every row the database holds — its
// rentals above all — survives.
func conform(db schemaDB, required shape) error {
	present, err := presentObjects(db)
	if err != nil {
		return err
	}
	for _, table := range required.tables {
		if !present[table.name] {
			if _, err := db.Exec(table.ddl); err != nil {
				return fmt.Errorf("create missing table %s: %w", table.name, err)
			}
			continue
		}
		have, err := tableColumns(db, table.name)
		if err != nil {
			return fmt.Errorf("inspect table %s: %w", table.name, err)
		}
		existing := make(map[string]bool, len(have))
		for _, column := range have {
			existing[strings.ToLower(column.name)] = true
		}
		for _, column := range table.columns {
			if existing[strings.ToLower(column.name)] {
				continue
			}
			if column.add == "" {
				return fmt.Errorf("table %s lacks its key column %s", table.name, column.name)
			}
			if _, err := db.Exec(fmt.Sprintf("ALTER TABLE %q ADD COLUMN %s", table.name, column.add)); err != nil {
				return fmt.Errorf("add missing column %s.%s: %w", table.name, column.name, err)
			}
		}
	}
	for _, object := range required.objects {
		if present[object.name] {
			continue
		}
		if _, err := db.Exec(object.ddl); err != nil {
			return fmt.Errorf("create missing %s %s: %w", object.kind, object.name, err)
		}
	}
	return nil
}

// missing names what a database lacks of the required shape, for a reader that may not
// change it.
func missing(db schemaDB, required shape) ([]string, error) {
	present, err := presentObjects(db)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, table := range required.tables {
		if !present[table.name] {
			out = append(out, "table "+table.name)
			continue
		}
		have, err := tableColumns(db, table.name)
		if err != nil {
			return nil, err
		}
		existing := make(map[string]bool, len(have))
		for _, column := range have {
			existing[strings.ToLower(column.name)] = true
		}
		for _, column := range table.columns {
			if !existing[strings.ToLower(column.name)] {
				out = append(out, "column "+table.name+"."+column.name)
			}
		}
	}
	return out, nil
}

func presentObjects(db schemaDB) (map[string]bool, error) {
	rows, err := db.Query(`SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	present := map[string]bool{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil, err
		}
		present[name] = true
	}
	return present, rows.Err()
}
