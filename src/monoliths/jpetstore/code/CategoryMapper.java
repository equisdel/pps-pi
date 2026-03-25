package org.mybatis.jpetstore.mapper;

import java.util.List;

import org.mybatis.jpetstore.domain.Category;

/**
 * The Interface CategoryMapper.
 *
 * @author Eduardo Macarron
 */
public interface CategoryMapper {

  List<Category> getCategoryList();

  Category getCategory(String categoryId);

}
